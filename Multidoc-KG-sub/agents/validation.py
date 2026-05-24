"""
知识验证代理：使用 LLM 判断检查与历史知识的逻辑冲突。
支持批量处理、缓存和并行验证，区分本体和实例。
"""
import json
import hashlib
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from schema import KnowledgeClaim, ClaimStatus
from core.llm_client import LLMClient
from core.graph_store import MockGraphStore
from core.neo4j_store import Neo4jGraphStore


class KnowledgeValidationAgent:
    """负责验证声明与现有图知识是否冲突的代理。"""
    
    def __init__(self, 
                 llm_client: LLMClient, 
                 graph_store,
                 use_cache: bool = True,
                 cache_size: int = 10000,
                 batch_size: int = 20,
                 max_workers: int = 8,
                 enable_parallel: bool = True,
                 skip_validation: bool = False):
        """
        初始化知识验证代理。
        
        Args:
            llm_client: 用于验证判断的 LLM 客户端
            graph_store: 用于查询历史声明的图数据库存储
            use_cache: 是否使用验证缓存
            cache_size: 最大缓存大小
            batch_size: 每批处理的声明数量
            max_workers: 最大并行工作线程数
            enable_parallel: 是否启用并行处理
            skip_validation: 是否跳过验证（所有声明直接通过）
        """
        self.llm_client = llm_client
        self.graph_store = graph_store
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        self.skip_validation = skip_validation
        
        # 每个 worker 线程独立持有一个 LLMClient，避免共享 httpx.Client 导致串行
        self._thread_local = threading.local()
        
        # 初始化缓存
        self.validation_cache = {} if use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 统计信息
        self.stats = {
            'total_validated': 0,
            'total_rejected': 0,
            'cache_hits': 0,
            'batch_count': 0,
            'api_calls': 0,
            'avg_confidence': 0.0
        }
        
        print(f"[KnowledgeValidationAgent] 初始化完成")
        if skip_validation:
            print(f"[KnowledgeValidationAgent]   模式: 跳过验证（所有声明直接通过）")
        else:
            print(f"[KnowledgeValidationAgent]   缓存: {'启用' if use_cache else '禁用'}")
        print(f"[KnowledgeValidationAgent]   批大小: {batch_size}")
        print(f"[KnowledgeValidationAgent]   并行: {enable_parallel} (工作线程={max_workers})")
    
    def _get_llm_client(self) -> LLMClient:
        """每个线程第一次调用时创建自己独立的 LLMClient 实例。"""
        if not hasattr(self._thread_local, 'client'):
            self._thread_local.client = LLMClient(
                model_name=self.llm_client.model_name,
                base_url=self.llm_client.base_url,
                timeout=self.llm_client.timeout,
            )
        return self._thread_local.client

    def _get_claim_hash(self, claim: KnowledgeClaim) -> str:
        """
        为声明生成唯一哈希。
        
        Args:
            claim: KnowledgeClaim 对象
            
        Returns:
            MD5 哈希字符串
        """
        key = f"{claim.subject}|{claim.relation}|{claim.object}|{claim.source_paper_id}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_claim_fingerprint(self, claim: KnowledgeClaim) -> str:
        """
        为相似性检测生成指纹（忽略证据）。
        
        Args:
            claim: KnowledgeClaim 对象
            
        Returns:
            用于相似性匹配的 MD5 哈希字符串
        """
        key = f"{claim.subject}|{claim.relation}|{claim.object}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _fast_filter(self, claims: List[KnowledgeClaim]) -> List[KnowledgeClaim]:
        """
        快速过滤掉明显的重复项。
        
        Args:
            claims: 声明列表
            
        Returns:
            过滤后的唯一声明列表
        """
        seen = set()
        unique = []
        
        for claim in claims:
            fingerprint = self._get_claim_fingerprint(claim)
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(claim)
        
        if len(unique) < len(claims):
            print(f"[KnowledgeValidationAgent] 快速过滤移除 {len(claims) - len(unique)} 个重复项")
        
        return unique
    
    def _is_valid_entity_name(self, entity_name: str) -> bool:
        """
        检查实体名称是否有效。

        Args:
            entity_name: 实体名称

        Returns:
            是否有效
        """
        if not entity_name or not entity_name.strip():
            return False

        # 文档结构词与明显噪声词
        blacklist_exact = [
            '示例', '食谱示例', '附录', '参考文献', '目录', '章节', '表格', '注释', '说明', '附表', '附件', '索引', '图表', '备注',
            'abstract', 'introduction', 'discussion', 'conclusion', 'conclusions',
            'results', 'methods', 'materials', 'keywords', 'figure', 'fig', 'table',
            'supplementary', 'supporting information', 'acknowledgements', 'funding',
            'references', 'graphical abstract', 'page'
        ]

        entity_lower = entity_name.lower()
        for word in blacklist_exact:
            if word in entity_lower:
                # 精确匹配或作为独立词出现
                if entity_name == word or \
                   entity_name.startswith(word) or \
                   entity_name.endswith(word):
                    return False

        return True

    def _get_claim_type_text(self, claim: KnowledgeClaim) -> str:
        """
        获取可读的声明类型文本。
        
        Args:
            claim: KnowledgeClaim 对象
            
        Returns:
            "本体" 或 "实例"
        """
        return "本体" if getattr(claim, 'claim_type', '') == "ontology" else "实例"

    def _build_claim_context(self, claim: KnowledgeClaim) -> str:
        """Build readable claim-level metadata context for prompts."""
        metadata = claim.metadata or {}
        context_lines = []

        paper_title = metadata.get("paper_title")
        if paper_title:
            context_lines.append(f"论文标题: {paper_title}")

        subdomain = metadata.get("paper_subdomain")
        if subdomain:
            context_lines.append(f"论文子领域: {subdomain}")

        parent_domain = metadata.get("paper_parent_domain")
        if parent_domain:
            context_lines.append(f"父领域: {parent_domain}")

        section_title = metadata.get("section_title")
        if section_title:
            context_lines.append(f"章节: {section_title}")

        return "\n".join(context_lines)

    def _apply_structural_checks(self, claim: KnowledgeClaim) -> bool:
        """
        对声明做结构性硬校验。

        Returns:
            True 表示已拒绝该声明；False 表示通过硬校验。
        """
        # 1. 自环检查（基于 grounded ID）
        if claim.subject_id and claim.object_id and claim.subject_id == claim.object_id:
            claim.status = ClaimStatus.REJECTED
            claim.confidence = 0.0
            claim.validation_type = "self_loop"
            self.stats['total_rejected'] += 1
            print(f"[KnowledgeValidationAgent] ✗ 拒绝（自环）: {claim.subject} -> {claim.object} (ID: {claim.subject_id})")
            return True

        # 2. 实体名称相同检查（防止 grounding 前的自环）
        if claim.subject.strip().lower() == claim.object.strip().lower():
            claim.status = ClaimStatus.REJECTED
            claim.confidence = 0.0
            claim.validation_type = "same_entity"
            self.stats['total_rejected'] += 1
            print(f"[KnowledgeValidationAgent] ✗ 拒绝（实体名称相同）: {claim.subject}")
            return True

        # 3. 实体质量检查
        if not self._is_valid_entity_name(claim.subject):
            claim.status = ClaimStatus.REJECTED
            claim.confidence = 0.0
            claim.validation_type = "invalid_subject"
            self.stats['total_rejected'] += 1
            print(f"[KnowledgeValidationAgent] ✗ 拒绝（主体实体无效）: {claim.subject}")
            return True

        if not self._is_valid_entity_name(claim.object):
            claim.status = ClaimStatus.REJECTED
            claim.confidence = 0.0
            claim.validation_type = "invalid_object"
            self.stats['total_rejected'] += 1
            print(f"[KnowledgeValidationAgent] ✗ 拒绝（客体实体无效）: {claim.object}")
            return True

        return False
    
    def _validate_claim(self, claim: KnowledgeClaim) -> KnowledgeClaim:
        """
        验证单个声明（用于并行处理）。

        Args:
            claim: 要验证的单个声明

        Returns:
            验证后的声明
        """
        try:
            # === 结构完整性检查 ===
            if self._apply_structural_checks(claim):
                return claim

            # 检查缓存
            if self.use_cache and self.validation_cache is not None:
                claim_hash = self._get_claim_hash(claim)
                if claim_hash in self.validation_cache:
                    cached = self.validation_cache[claim_hash]
                    claim.status = cached['status']
                    claim.confidence = cached['confidence']
                    claim.validation_type = cached.get('type', 'new')
                    self.cache_hits += 1
                    self.stats['cache_hits'] += 1
                    print(f"[KnowledgeValidationAgent] ✓ 缓存命中: {claim.subject} {claim.relation} {claim.object[:30]}...")
                    return claim
                self.cache_misses += 1
            
            claim_type = self._get_claim_type_text(claim)
            claim_str = f"{claim.subject} {claim.relation} {claim.object}"
            claim_context = self._build_claim_context(claim)
            
            # 获取历史上下文
            historical_context = []
            if claim.subject:
                historical_context = self.graph_store.get_entity_context(claim.subject)
            
            system_prompt = (
                "你是一名医学文献知识图谱验证专家。\n"
                "你的任务是判断新提取的 biomedical knowledge claim 是否应被接受。\n\n"
                "注意区分：\n"
                "- 本体层（ontology）：概念层知识，如疾病分类、药物类型、biomarker 关系\n"
                "- 实例层（instance）：论文/病例/实验层知识，如患者诊断、治疗、队列观察结果\n\n"
                "验证原则：\n"
                "- 优先依据证据原文判断，不要臆测证据外信息\n"
                "- 如果与已有知识明显一致，标记为 support\n"
                "- 如果与已有知识明显冲突，标记为 conflict\n"
                "- 如果证据支持、但图中没有类似知识，可标记为 new\n"
                "- 如果证据不足、关系不清、实体明显异常，应拒绝\n\n"
                "只返回JSON，不要其他文字。"
            )
            
            user_prompt = f"新知识 [{claim_type}]: {claim_str}\n"
            user_prompt += f"证据原文: {claim.evidence[:200]}...\n\n"

            if claim_context:
                user_prompt += f"论文上下文:\n{claim_context}\n\n"
            
            if historical_context:
                context_str = "\n".join([f"  - {ctx}" for ctx in historical_context[:5]])
                user_prompt += f"已有知识:\n{context_str}\n\n"
                user_prompt += "任务：检查与已有知识是否存在逻辑冲突。\n"
            else:
                user_prompt += "任务：评估该知识是否被论文证据支持，并适合作为医学知识图谱中的新知识保留。\n"
            
            user_prompt += """
返回格式：
{
  "valid": boolean,      # true表示有效，false表示无效
  "type": "support" | "conflict" | "new",  # 支持/冲突/新知识
  "confidence": float,    # 置信度0-1
  "reasoning": string     # 简短理由（用中文）
}"""
            
            print(f"[KnowledgeValidationAgent] 验证 [{claim_type}]: '{claim_str[:60]}...'")
            
            response = self._get_llm_client().generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                max_tokens=1000
            )
            
            self.stats['api_calls'] += 1
            
            # 解析响应
            if isinstance(response, str):
                response_clean = response.strip()
                if response_clean.startswith("```json"):
                    response_clean = response_clean[7:]
                elif response_clean.startswith("```"):
                    response_clean = response_clean[3:]
                if response_clean.endswith("```"):
                    response_clean = response_clean[:-3]
                response_clean = response_clean.strip()
                
                judgment = json.loads(response_clean)
            else:
                judgment = response
            
            # 处理列表情况
            if isinstance(judgment, list) and judgment:
                judgment = judgment[0]
            
            if isinstance(judgment, dict):
                is_valid = judgment.get("valid", False)
                validation_type = judgment.get("type", "new")
                confidence = judgment.get("confidence", 0.5)
                reasoning = judgment.get("reasoning", "")
                
                claim.validation_type = validation_type
                claim.confidence = confidence
                
                if is_valid:
                    claim.status = ClaimStatus.VALIDATED
                    self.stats['total_validated'] += 1
                    print(f"[KnowledgeValidationAgent] ✓ 验证通过: 类型={validation_type}, 置信度={confidence:.2f}")
                else:
                    claim.status = ClaimStatus.REJECTED
                    self.stats['total_rejected'] += 1
                    print(f"[KnowledgeValidationAgent] ✗ 验证拒绝: {reasoning[:80]}...")
            else:
                # LLM响应格式无效，保守拒绝
                claim.status = ClaimStatus.REJECTED
                claim.confidence = 0.0
                claim.validation_type = "format_error"
                self.stats['total_rejected'] += 1
                print(f"[KnowledgeValidationAgent] ⚠ 拒绝（响应格式无效）")
            
            # 更新缓存
            if self.use_cache and self.validation_cache is not None:
                if len(self.validation_cache) < self.cache_size:
                    claim_hash = self._get_claim_hash(claim)
                    self.validation_cache[claim_hash] = {
                        'status': claim.status,
                        'confidence': claim.confidence,
                        'type': claim.validation_type
                    }
            
        except Exception as e:
            print(f"[KnowledgeValidationAgent] 验证声明时出错: {str(e)}")
            # 出错时默认拒绝，保证知识图谱质量
            claim.status = ClaimStatus.REJECTED
            claim.confidence = 0.0
            claim.validation_type = "error"
            self.stats['total_rejected'] += 1

        return claim
    
    def _batch_validate(self, claims: List[KnowledgeClaim]) -> List[KnowledgeClaim]:
        """
        批量验证一批声明以减少 API 调用。
        
        Args:
            claims: 要验证的一批声明
            
        Returns:
            验证后的声明列表
        """
        if not claims:
            return []

        # 批量路径也必须先做硬校验，避免绕过自环过滤
        pre_rejected = []
        to_validate = []
        for claim in claims:
            if self._apply_structural_checks(claim):
                pre_rejected.append(claim)
            else:
                to_validate.append(claim)

        if not to_validate:
            return pre_rejected

        claims = to_validate
        
        # 构建批量验证的提示词，区分本体和实例
        claim_list = []
        for i, claim in enumerate(claims):
            claim_type = self._get_claim_type_text(claim)
            claim_list.append(f"声明 {i+1} [{claim_type}]: {claim.subject} {claim.relation} {claim.object}")
            claim_list.append(f"证据 {i+1}: {claim.evidence[:100]}...")
            claim_context = self._build_claim_context(claim)
            if claim_context:
                claim_list.append(f"上下文 {i+1}: {claim_context}")
        
        claims_text = "\n".join(claim_list)
        
        # 批量获取上下文（单次 Neo4j 查询替代串行循环）
        subjects = [c.subject for c in claims if c.subject]
        # 注意：需要确保 graph_store 实现了 get_entity_contexts_batch 方法
        # 如果没有，可以回退到串行查询，或实现批量查询
        try:
            contexts_map = self.graph_store.get_entity_contexts_batch(subjects)
        except AttributeError:
            # 回退到串行查询
            contexts_map = {}
            for subject in subjects:
                ctx = self.graph_store.get_entity_context(subject)
                if ctx:
                    contexts_map[subject] = ctx
        
        all_contexts = []
        for claim in claims:
            ctx = contexts_map.get(claim.subject, [])
            if ctx:
                all_contexts.extend(ctx[:3])
        
        system_prompt = (
            "你是一名医学文献知识图谱验证专家。\n"
            "你的任务是批量判断多个新提取的 biomedical knowledge claims 是否应被接受。\n\n"
            "注意区分两类知识：\n"
            "1. 本体层（ontology）：定义概念及其关系，如疾病亚型、药物类型、biomarker 关系\n"
            "   - 验证概念层次与关系方向是否合理\n"
            "2. 实例层（instance）：描述论文/病例/队列/实验中的具体关系\n"
            "   - 验证观察结果、治疗关系、诊断关系是否被证据支持\n\n"
            "重要原则：\n"
            "- 优先依据证据原文，不要补全未明说的信息\n"
            "- 若与图谱已有知识一致，返回 support\n"
            "- 若与已有知识明显冲突，返回 conflict\n"
            "- 若证据支持但图谱暂无类似知识，可返回 new\n"
            "- 若证据不足、实体异常、关系不清，应拒绝\n\n"
            "返回JSON数组，每个元素对应一个声明的判断。"
        )
        
        context_text = ""
        if all_contexts:
            context_text = "\n已有知识图谱中的相关信息：\n" + "\n".join([f"  - {ctx}" for ctx in all_contexts[:10]])
        
        user_prompt = f"""请验证以下{len(claims)}个医学知识声明（已标注类型）：

{claims_text}
{context_text}

为每个声明返回一个JSON对象，包含：
- valid: boolean (是否有效)
- type: "support" | "conflict" | "new" (支持/冲突/新知识)
- confidence: float (0-1)
- reasoning: string (简短理由，用中文)

返回格式必须是JSON数组，按声明顺序对应：
[
  {{"valid": true, "type": "support", "confidence": 0.95, "reasoning": "证据支持且与已有知识一致..."}},
  {{"valid": false, "type": "conflict", "confidence": 0.8, "reasoning": "与已有知识冲突..."}}
]"""

        try:
            print(f"[KnowledgeValidationAgent] 批量验证 {len(claims)} 个声明...")
            
            response = self._get_llm_client().generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                max_tokens=4000
            )
            
            self.stats['api_calls'] += 1
            
            # 解析批量结果
            if isinstance(response, str):
                # 清理可能的markdown
                response_clean = response.strip()
                if response_clean.startswith("```json"):
                    response_clean = response_clean[7:]
                elif response_clean.startswith("```"):
                    response_clean = response_clean[3:]
                if response_clean.endswith("```"):
                    response_clean = response_clean[:-3]
                response_clean = response_clean.strip()
                
                judgments = json.loads(response_clean)
            else:
                judgments = response
            
            # 确保是列表
            if isinstance(judgments, dict) and "judgments" in judgments:
                judgments = judgments["judgments"]
            elif isinstance(judgments, dict):
                judgments = [judgments]
            
            # 应用判断结果
            validated_batch = []
            for i, claim in enumerate(claims):
                if i < len(judgments):
                    judgment = judgments[i]
                    if isinstance(judgment, dict):
                        is_valid = judgment.get("valid", False)
                        validation_type = judgment.get("type", "unknown")
                        confidence = judgment.get("confidence", 0.5)
                        reasoning = judgment.get("reasoning", "")
                        
                        claim.validation_type = validation_type
                        claim.confidence = confidence
                        
                        if is_valid:
                            claim.status = ClaimStatus.VALIDATED
                            self.stats['total_validated'] += 1
                            print(f"[KnowledgeValidationAgent] ✓ 批量 {i+1} 验证通过: {validation_type}, {confidence:.2f}")
                        else:
                            claim.status = ClaimStatus.REJECTED
                            self.stats['total_rejected'] += 1
                            print(f"[KnowledgeValidationAgent] ✗ 批量 {i+1} 拒绝: {reasoning[:50]}...")
                    else:
                        # 批量响应格式无效，保守拒绝
                        claim.status = ClaimStatus.REJECTED
                        claim.confidence = 0.0
                        claim.validation_type = "format_error"
                        self.stats['total_rejected'] += 1
                        print(f"[KnowledgeValidationAgent] ⚠ 批量 {i+1} 拒绝（判断格式无效）")
                else:
                    # 批量响应中无此条目，保守拒绝
                    claim.status = ClaimStatus.REJECTED
                    claim.confidence = 0.0
                    claim.validation_type = "format_error"
                    self.stats['total_rejected'] += 1
                    print(f"[KnowledgeValidationAgent] ⚠ 批量 {i+1} 拒绝（无对应判断）")
                
                validated_batch.append(claim)
            
            print(f"[KnowledgeValidationAgent] 批量完成: 处理 {len(validated_batch)} 个声明")
            return pre_rejected + validated_batch
            
        except Exception as e:
            print(f"[KnowledgeValidationAgent] 批量验证错误: {str(e)}")
            # 出错时默认拒绝所有，保证知识图谱质量
            for claim in claims:
                claim.status = ClaimStatus.REJECTED
                claim.confidence = 0.0
                claim.validation_type = "error"
                self.stats['total_rejected'] += 1
            return pre_rejected + claims
    
    def process(self, claims: List[KnowledgeClaim]) -> List[KnowledgeClaim]:
        """
        使用优化的批处理和并行处理验证声明。
        
        Args:
            claims: 已接地的 KnowledgeClaim 对象列表
            
        Returns:
            验证通过或拒绝的 KnowledgeClaim 对象列表
        """
        print(f"\n[KnowledgeValidationAgent] 处理 {len(claims)} 个声明")

        # 跳过模式：跳过 LLM，全部标记为 VALIDATED
        if self.skip_validation:
            for claim in claims:
                claim.status = ClaimStatus.VALIDATED
                claim.confidence = 1.0
                claim.validation_type = "skip"
            self.stats['total_validated'] = len(claims)
            print(f"[KnowledgeValidationAgent] 跳过模式: 全部 {len(claims)} 个声明标记为通过")
            return claims

        if not claims:
            return []
        
        # 1. 快速去重
        unique_claims = self._fast_filter(claims)
        
        # 2. 检查缓存
        if self.use_cache and self.validation_cache is not None:
            cached_claims = []
            to_validate = []
            for claim in unique_claims:
                claim_hash = self._get_claim_hash(claim)
                if claim_hash in self.validation_cache:
                    cached = self.validation_cache[claim_hash]
                    claim.status = cached['status']
                    claim.confidence = cached['confidence']
                    claim.validation_type = cached.get('type', 'new')
                    cached_claims.append(claim)
                    self.cache_hits += 1
                    self.stats['cache_hits'] += 1
                else:
                    to_validate.append(claim)
            
            print(f"[KnowledgeValidationAgent] 缓存命中: {len(cached_claims)}，待验证: {len(to_validate)}")
            unique_claims = to_validate
        
        if not unique_claims:
            print("[KnowledgeValidationAgent] 所有声明来自缓存")
            return cached_claims
        
        # 3. 按类型分组统计
        ontology_count = sum(1 for c in unique_claims if getattr(c, 'claim_type', '') == "ontology")
        instance_count = len(unique_claims) - ontology_count
        print(f"[KnowledgeValidationAgent] 待验证: {ontology_count} 本体, {instance_count} 实例")
        
        # 4. 决定使用批量验证还是单条验证
        validated_claims = []
        
        if len(unique_claims) >= self.batch_size:
            # 批量验证模式
            batches = [unique_claims[i:i+self.batch_size] for i in range(0, len(unique_claims), self.batch_size)]
            self.stats['batch_count'] += len(batches)
            
            print(f"[KnowledgeValidationAgent] 拆分为 {len(batches)} 批，每批约 {self.batch_size} 个")
            
            if self.enable_parallel and len(batches) > 1:
                # 并行处理批次
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_batch = {executor.submit(self._batch_validate, batch): i for i, batch in enumerate(batches)}
                    
                    for future in as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_result = future.result()
                            validated_claims.extend(batch_result)
                            print(f"[KnowledgeValidationAgent] 批 {batch_idx+1}/{len(batches)} 完成")
                        except Exception as e:
                            print(f"[KnowledgeValidationAgent] 批 {batch_idx+1} 失败: {str(e)}")
                            # 失败时接受该批次
                            for claim in batches[batch_idx]:
                                claim.status = ClaimStatus.VALIDATED
                                claim.confidence = 0.5
                                claim.validation_type = "new"
                                validated_claims.append(claim)
            else:
                # 串行处理批次
                for i, batch in enumerate(batches):
                    print(f"[KnowledgeValidationAgent] 处理批 {i+1}/{len(batches)}")
                    batch_result = self._batch_validate(batch)
                    validated_claims.extend(batch_result)
        else:
            # 单条验证模式（少量声明时使用）
            print(f"[KnowledgeValidationAgent] 使用单条验证模式处理 {len(unique_claims)} 个声明")
            
            if self.enable_parallel and len(unique_claims) > 1:
                # 并行验证单条声明
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 使用 map 进行并行处理
                    validated_claims = list(executor.map(self._validate_claim, unique_claims))
            else:
                # 串行验证
                for claim in unique_claims:
                    validated_claims.append(self._validate_claim(claim))
        
        # 5. 合并缓存结果
        if self.use_cache and self.validation_cache is not None:
            validated_claims.extend(cached_claims)
        
        # 6. 更新统计
        validated_count = sum(1 for c in validated_claims if c.status == ClaimStatus.VALIDATED)
        rejected_count = len(validated_claims) - validated_count
        
        self.stats['total_validated'] = validated_count
        self.stats['total_rejected'] = rejected_count
        
        # 7. 按类型统计
        validated_ontology = sum(1 for c in validated_claims if c.status == ClaimStatus.VALIDATED and getattr(c, 'claim_type', '') == "ontology")
        validated_instance = validated_count - validated_ontology
        
        print(f"[KnowledgeValidationAgent] 最终结果: {validated_count} 通过 ({validated_ontology} 本体, {validated_instance} 实例), {rejected_count} 拒绝")
        print(f"[KnowledgeValidationAgent]   API 调用: {self.stats['api_calls']}, 缓存命中: {self.cache_hits}, 命中率: {self.cache_hits/(self.cache_hits+self.cache_misses+0.001):.2%}")
        
        return validated_claims
    
    def get_stats(self) -> Dict[str, Any]:
        """获取验证统计信息。"""
        self.stats['cache_hits'] = self.cache_hits
        self.stats['cache_misses'] = self.cache_misses
        self.stats['cache_hit_rate'] = self.cache_hits / (self.cache_hits + self.cache_misses + 0.001)
        return self.stats
    
    def clear_cache(self):
        """清除验证缓存。"""
        if self.validation_cache is not None:
            self.validation_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            print("[KnowledgeValidationAgent] 缓存已清除")
