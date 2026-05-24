"""
语义接地代理：使用向量搜索 + LLM 判断将实体与现有图谱对齐。
支持概念（本体）和实例（实体）的区分。
"""
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from schema import KnowledgeClaim, ClaimStatus
from core.llm_client import LLMClient
from core.vector_store import VectorStore


class SemanticGroundingAgent:
    """负责将实体与现有图节点进行接地的代理。"""
    
    def __init__(self, llm_client: LLMClient, vector_store: VectorStore):
        """
        初始化语义接地代理。
        
        Args:
            llm_client: 用于实体接地判断的 LLM 客户端
            vector_store: 用于语义实体搜索的向量存储
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        
        # 实体解析缓存：实体名称小写 -> 实体ID
        self._entity_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_entities': 0,        # 处理的实体总数
            'concept_matches': 0,        # 概念匹配数
            'instance_matches': 0,       # 实例匹配数
            'new_concepts': 0,           # 新创建的概念数
            'new_instances': 0,           # 新创建的实例数
            'llm_calls': 0,               # LLM调用次数
            'cache_hits': 0                # 缓存命中次数
        }
        
        print("[SemanticGroundingAgent] 初始化完成（带缓存和并行处理）")

    def _build_entity_context(self, claim: KnowledgeClaim) -> str:
        """Build biomedical grounding context from claim evidence and metadata."""
        metadata = claim.metadata or {}
        parts = [f"证据: {claim.evidence[:200]}..."]

        paper_title = metadata.get("paper_title")
        if paper_title:
            parts.append(f"论文标题: {paper_title}")

        subdomain = metadata.get("paper_subdomain")
        if subdomain:
            parts.append(f"论文子领域: {subdomain}")

        section_title = metadata.get("section_title")
        if section_title:
            parts.append(f"章节: {section_title}")

        if getattr(claim, 'claim_type', ''):
            parts.append(f"声明类型: {claim.claim_type}")

        return " | ".join(parts)
    
    def _resolve_entity(self, entity_name: str, context: str = "", is_concept: bool = False) -> str:
        """
        使用向量搜索 + LLM 判断将实体解析为接地 ID。
        
        策略：
        1. 缓存：检查实体是否已解析
        2. 检索：在向量存储中搜索相似实体
        3. 精确匹配：如果找到精确匹配（忽略大小写），返回现有 ID
        4. LLM 判断：如果有候选，用 LLM 决定合并还是新建
        5. 新建：如果没有匹配，生成新 ID 并添加到向量存储
        
        Args:
            entity_name: 要解析的实体名称
            context: 关于实体的可选上下文
            is_concept: 该实体是概念（本体）还是实例
            
        Returns:
            接地的实体 ID（现有或新建）
        """
        entity_type = "概念" if is_concept else "实例"
        
        # 1. 命中缓存直接返回，跳过 FAISS 和 LLM
        cache_key = entity_name.lower()
        with self._cache_lock:
            if cache_key in self._entity_cache:
                print(f"[Grounding] 缓存命中: '{entity_name}' -> '{self._entity_cache[cache_key]}'")
                self.stats['cache_hits'] += 1
                return self._entity_cache[cache_key]
        
        print(f"[Grounding] 解析 {entity_type}: '{entity_name}'")
        
        # 2. 检索 - 搜索相似实体
        candidates = self.vector_store.search(entity_name, top_k=5)
        
        # 3. 根据类型设置不同的匹配阈值和策略
        resolved_id = None
        
        if is_concept:
            # 概念匹配：概念名称通常更规范
            # 先检查精确匹配
            for candidate in candidates:
                if candidate['name'].lower() == entity_name.lower():
                    print(f"[Grounding] 精确概念匹配: '{entity_name}' -> '{candidate['name']}' (ID: {candidate['id']})")
                    resolved_id = candidate['id']
                    self.stats['concept_matches'] += 1
                    break
            
            if not resolved_id:
                # 防止子串误合并：过滤掉长度差异大的候选
                filtered_candidates = []
                for candidate in candidates:
                    # 如果实体名是候选的子串或超串，但长度差异大，跳过
                    if (entity_name in candidate['name'] or candidate['name'] in entity_name):
                        len_diff = abs(len(entity_name) - len(candidate['name']))
                        if len_diff > 3:
                            print(f"[Grounding] 跳过子串候选（长度差异={len_diff}）: '{entity_name}' vs '{candidate['name']}'")
                            continue
                    filtered_candidates.append(candidate)

                # 概念匹配阈值（L2距离 < 1.0），避免图谱碎片化
                score_threshold = 1.0
                good_candidates = [c for c in filtered_candidates if c['score'] < score_threshold]
                
                # 如果只有一个很好的候选，直接采用
                if len(good_candidates) == 1 and good_candidates[0]['score'] < 0.7:
                    print(f"[Grounding] 强概念匹配: '{entity_name}' -> '{good_candidates[0]['name']}' (ID: {good_candidates[0]['id']})")
                    resolved_id = good_candidates[0]['id']
                    self.stats['concept_matches'] += 1
                elif good_candidates:
                    # 使用 LLM 判断
                    decision = self._llm_judge_entity(entity_name, good_candidates, context, is_concept)
                    if decision['decision'] == 'merge' and 'id' in decision:
                        resolved_id = decision['id']
                        merged_candidate = next((c for c in good_candidates if c['id'] == resolved_id), None)
                        if merged_candidate:
                            print(f"[Grounding] LLM 合并概念: '{entity_name}' -> '{merged_candidate['name']}' (ID: {resolved_id})")
                        self.stats['concept_matches'] += 1
        else:
            # 实例匹配：实例名称可能更多样，需要更严格的匹配
            # 检查精确匹配
            for candidate in candidates:
                if candidate['name'].lower() == entity_name.lower():
                    print(f"[Grounding] 精确实例匹配: '{entity_name}' -> '{candidate['name']}' (ID: {candidate['id']})")
                    resolved_id = candidate['id']
                    self.stats['instance_matches'] += 1
                    break
            
            if not resolved_id:
                # 防止子串误合并：过滤掉长度差异大的候选
                filtered_candidates = []
                for candidate in candidates:
                    # 如果实体名是候选的子��或超串，但长度差异大，跳过
                    if (entity_name in candidate['name'] or candidate['name'] in entity_name):
                        len_diff = abs(len(entity_name) - len(candidate['name']))
                        if len_diff > 3:
                            print(f"[Grounding] 跳过子串候选（长度差异={len_diff}）: '{entity_name}' vs '{candidate['name']}'")
                            continue
                    filtered_candidates.append(candidate)

                # 实例匹配阈值（L2距离 < 1.0）
                score_threshold = 1.0
                good_candidates = [c for c in filtered_candidates if c['score'] < score_threshold]
                
                # 如果只有一个很好的候选，直接采用
                if len(good_candidates) == 1 and good_candidates[0]['score'] < 0.7:
                    print(f"[Grounding] 强实例匹配: '{entity_name}' -> '{good_candidates[0]['name']}' (ID: {good_candidates[0]['id']})")
                    resolved_id = good_candidates[0]['id']
                    self.stats['instance_matches'] += 1
                elif good_candidates:
                    # 使用 LLM 判断
                    decision = self._llm_judge_entity(entity_name, good_candidates, context, is_concept)
                    if decision['decision'] == 'merge' and 'id' in decision:
                        resolved_id = decision['id']
                        merged_candidate = next((c for c in good_candidates if c['id'] == resolved_id), None)
                        if merged_candidate:
                            print(f"[Grounding] LLM 合并实例: '{entity_name}' -> '{merged_candidate['name']}' (ID: {resolved_id})")
                        self.stats['instance_matches'] += 1
        
        # 4. 如果找到了匹配的ID，更新缓存并返回
        if resolved_id:
            with self._cache_lock:
                self._entity_cache[cache_key] = resolved_id
            self.stats['total_entities'] += 1
            return resolved_id
        
        # 5. 没有找到匹配，创建新实体
        new_id = f"ent_{uuid.uuid4().hex[:16]}"
        self.vector_store.add_entity(entity_name, new_id)
        
        if is_concept:
            self.stats['new_concepts'] += 1
            print(f"[Grounding] 创建新概念: '{entity_name}' (ID: {new_id})")
        else:
            self.stats['new_instances'] += 1
            print(f"[Grounding] 创建新实例: '{entity_name}' (ID: {new_id})")
        
        with self._cache_lock:
            self._entity_cache[cache_key] = new_id
        self.stats['total_entities'] += 1
        return new_id
    
    def _llm_judge_entity(self, target_entity: str, candidates: List[Dict], context: str = "", is_concept: bool = False) -> Dict[str, str]:
        """
        使用 LLM 判断目标实体是否应与现有候选合并。
        
        Args:
            target_entity: 要接地的目标实体
            candidates: 向量搜索的候选实体列表
            context: 关于实体的可选上下文
            is_concept: 该实体是否是概念
            
        Returns:
            包含 'decision' ('merge' 或 'new') 和可选的 'id' 的字典
        """
        self.stats['llm_calls'] += 1
        
        # 构建候选列表用于提示词
        candidate_text = "\n".join([
            f"  - '{c['name']}' (ID: {c['id']}, 相似度得分: {c['score']:.3f})"
            for c in candidates
        ])
        
        entity_type = "概念" if is_concept else "实例"
        
        system_prompt = (
            "你是一名医学知识图谱实体解析专家。\n"
            f"你的任务是判断一个目标{entity_type}是否应该与已有的{entity_type}合并，还是创建新的。\n"
            "只返回JSON格式，不要其他文字。"
        )
        
        if is_concept:
            # 概念合并的提示词
            user_prompt = (
                f"目标{entity_type}: '{target_entity}'\n\n"
                f"候选{entity_type}列表:\n{candidate_text}\n\n"
            )
            
            if context:
                user_prompt += f"上下文信息: {context}\n\n"
            
            user_prompt += (
                "这是一个医学概念，如疾病、亚型、药物、biomarker、检测方法、治疗策略等抽象范畴。\n"
                "概念合并的原则：\n"
                "1. 只有当两个概念完全等同或标准名称与别名关系时才合并\n"
                "2. 不同层次的概念不能合并（如：'liver cancer' 和 'hepatocellular carcinoma' 可能是上下位，不应直接视为同义）\n"
                "3. 常见缩写与全称、标准别名可以合并（如必要时结合上下文）\n"
                "4. 考虑医学术语的规范性与粒度\n"
                "5. **禁止合并规则**：\n"
                "   - 如果目标实体包含'示例'、'食谱'、'附录'、'参考文献'等文档结构词，必须创建新实体\n"
                "   - 如果目标实体是候选实体的子串或超串，但语义完全不同，必须创建新实体\n"
                "   - 例如：'高血压患者食谱示例' 不能与 '高血压' 合并\n"
                "   - 例如：'PD-L1 expression level' 不能直接与 'PD-L1' 合并，除非上下文明确是同一概念\n\n"
                "问题：目标概念是否与某个候选概念是同一个概念？\n\n"
                "返回JSON格式：\n"
                "- 如果是同一个概念：{\"decision\": \"merge\", \"id\": \"<候选ID>\", \"reasoning\": \"合并理由\"}\n"
                "- 如果是不同概念：{\"decision\": \"new\", \"reasoning\": \"创建新概念的理由\"}\n"
            )
        else:
            # 实例合并的提示词
            user_prompt = (
                f"目标{entity_type}: '{target_entity}'\n\n"
                f"候选{entity_type}列表:\n{candidate_text}\n\n"
            )
            
            if context:
                user_prompt += f"上下文信息: {context}\n\n"
            
            user_prompt += (
                "这是一个医学实例，如具体患者、队列、样本、细胞系、实验组、病例事件等具体实体。\n"
                "实例合并的原则：\n"
                "1. 只有当两个实例指向同一个具体实体时才合并\n"
                "2. 考虑上下文信息判断是否同一实体\n"
                "3. 不同患者、不同队列、不同样本即使标签相似也不能合并\n"
                "4. 仅当上下文支持同一实例时才合并；泛称如 'patient'、'cohort' 通常应新建或保持分离\n"
                "5. **禁止合并规则**：\n"
                "   - 如果目标实体包含'示例'、'食谱'、'附录'等文档结构词，必须创建新实体\n"
                "   - 如果目标实体是候选实体的子串或超串，但语义完全不同，必须创建新实体\n"
                "   - 例如：'高血压患者食谱示例' 不能与 '高血压' 合并\n\n"
                "问题：目标实例是否与某个候选实例是同一个实体？\n\n"
                "返回JSON格式：\n"
                "- 如果是同一个实体：{\"decision\": \"merge\", \"id\": \"<候选ID>\", \"reasoning\": \"合并理由\"}\n"
                "- 如果是不同实体：{\"decision\": \"new\", \"reasoning\": \"创建新实例的理由\"}\n"
            )
        
        try:
            response_text = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
            
            print(f"[Grounding] LLM 判断响应 for '{target_entity}': {response_text[:150]}...")
            
            # 清理响应
            if isinstance(response_text, list):
                response_text = json.dumps(response_text)
            elif not isinstance(response_text, str):
                response_text = str(response_text)
            
            response_clean = response_text.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            elif response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            # 解析 JSON
            try:
                parsed = json.loads(response_clean)
                
                # 处理列表响应
                if isinstance(parsed, list):
                    if len(parsed) > 0 and isinstance(parsed[0], dict):
                        decision = parsed[0]
                    else:
                        print(f"[Grounding] 警告: 空的或无效的列表响应")
                        return {"decision": "new", "reasoning": "LLM返回格式无效"}
                elif isinstance(parsed, dict):
                    decision = parsed
                else:
                    print(f"[Grounding] 警告: 意外的响应类型 {type(parsed)}")
                    return {"decision": "new", "reasoning": "返回类型错误"}
                
                # 验证决策
                if decision.get("decision") not in ["merge", "new"]:
                    print(f"[Grounding] 警告: 无效的决策值: {decision.get('decision')}")
                    return {"decision": "new", "reasoning": "决策值无效"}
                
                return decision
                
            except json.JSONDecodeError as e:
                print(f"[Grounding] 警告: 无法解析 JSON 响应: {str(e)}")
                print(f"[Grounding] 响应预览: {response_clean[:200]}...")
                return {"decision": "new", "reasoning": f"JSON解析错误"}
                
        except Exception as e:
            print(f"[Grounding] LLM 判断错误: {str(e)}")
            return {"decision": "new", "reasoning": f"LLM调用错误"}
    
    def _warm_up_cache(self, claims: List[KnowledgeClaim]) -> None:
        """
        批量预编码所有唯一实体名，然后填充实体缓存。
        这样后续的 _resolve_entity 调用几乎全部走缓存路径，跳过编码。
        
        对每个实体：
          - 如果已在 FAISS 中（精确匹配或强相似匹配），将现有 ID 存入缓存
          - 否则，批量添加到 FAISS 并将新 ID 存入缓存
        """
        if not hasattr(self.vector_store, "model") or self.vector_store.model is None:
            print("[Grounding] 当前向量存储不支持批量预热，跳过")
            return
        if not hasattr(self.vector_store, "index") or self.vector_store.index is None:
            print("[Grounding] 当前向量存储无索引对象，跳过批量预热")
            return

        # 收集所有唯一实体名（去重，保留大小写原始形式）
        unique_names: Dict[str, bool] = {}  # 小写名称 -> 是否为概念
        for claim in claims:
            is_concept = (getattr(claim, 'claim_type', '') == "ontology")
            unique_names.setdefault(claim.subject.lower(), is_concept)
            unique_names.setdefault(claim.object.lower(), is_concept)
        
        # 过滤掉已有缓存的
        with self._cache_lock:
            to_resolve = {k: v for k, v in unique_names.items() if k not in self._entity_cache}
        
        if not to_resolve:
            print(f"[Grounding] 所有 {len(unique_names)} 个实体已在缓存中，跳过预热")
            return
        
        names_list = list(to_resolve.keys())
        print(f"[Grounding] 批量预热: 编码 {len(names_list)} 个唯一实体...")
        
        # 一次性批量编码
        embeddings = self.vector_store.model.encode(
            names_list, convert_to_numpy=True, batch_size=256
        ).astype('float32')
        
        # 对每个实体决定是合并还是新建
        to_add_batch = []  # 需要新增到 FAISS 的实体
        for name, emb in zip(names_list, embeddings):
            # 在已有 FAISS 中搜索
            if self.vector_store.index.ntotal > 0:
                import numpy as np
                distances, indices = self.vector_store.index.search(
                    emb.reshape(1, -1), min(3, self.vector_store.index.ntotal)
                )
                best_dist = float(distances[0][0])
                best_idx = int(indices[0][0])
                best_meta = self.vector_store.metadata[best_idx] if best_idx < len(self.vector_store.metadata) else None
                
                # 精确匹配
                if best_meta and best_meta['name'].lower() == name:
                    with self._cache_lock:
                        self._entity_cache[name] = best_meta['id']
                    continue
                
                # 强相似匹配（L2 < 0.8）
                if best_dist < 0.8 and best_meta:
                    with self._cache_lock:
                        self._entity_cache[name] = best_meta['id']
                    continue
            
            # 需要新建实体
            new_id = f"ent_{uuid.uuid4().hex[:16]}"
            to_add_batch.append({'name': name, 'id': new_id, 'emb': emb})
            with self._cache_lock:
                self._entity_cache[name] = new_id
        
        # 批量写入 FAISS
        if to_add_batch:
            import numpy as np
            emb_matrix = np.stack([e['emb'] for e in to_add_batch]).astype('float32')
            self.vector_store.index.add(emb_matrix)
            self.vector_store.metadata.extend(
                [{'name': e['name'], 'id': e['id']} for e in to_add_batch]
            )
            print(f"[Grounding] 批量预热完成: 向 FAISS 添加 {len(to_add_batch)} 个新实体")
        else:
            print(f"[Grounding] 批量预热完成: 所有实体匹配现有条目")

    def process(self, claims: List[KnowledgeClaim]) -> List[KnowledgeClaim]:
        """
        将声明中的实体接地到现有图节点。
        
        Args:
            claims: 需要接地的 KnowledgeClaim 对象列表
            
        Returns:
            已接地的 KnowledgeClaim 对象列表（包含更新的实体 ID）
        """
        print(f"\n[SemanticGroundingAgent] 处理 {len(claims)} 个声明的接地")
        
        ontology_claims = sum(1 for c in claims if getattr(c, 'claim_type', '') == "ontology")
        instance_claims = len(claims) - ontology_claims
        print(f"[SemanticGroundingAgent] 声明统计: {ontology_claims} 本体, {instance_claims} 实例")
        
        # 批量预编码所有实体，填充缓存，避免逐条编码
        self._warm_up_cache(claims)
        
        def _ground_claim(claim):
            """处理单个声明的接地"""
            try:
                context = self._build_entity_context(claim)
                
                is_subject_concept = (getattr(claim, 'claim_type', '') == "ontology")
                is_object_concept = (getattr(claim, 'claim_type', '') == "ontology")
                
                print(f"\n[Grounding] 处理声明: {claim.subject} {claim.relation} {claim.object}")
                
                # 并行解析 subject 和 object
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_subj = executor.submit(
                        self._resolve_entity, 
                        claim.subject, 
                        context, 
                        is_concept=is_subject_concept
                    )
                    future_obj = executor.submit(
                        self._resolve_entity, 
                        claim.object, 
                        context, 
                        is_concept=is_object_concept
                    )
                    claim.subject_id = future_subj.result()
                    claim.object_id = future_obj.result()
                
                # 更新 grounded_ids 以保持向后兼容
                claim.grounded_ids = [claim.subject_id, claim.object_id]
                
                # 更新状态
                claim.status = ClaimStatus.GROUNDED
                
                return claim
                
            except Exception as e:
                print(f"[Grounding] 接地声明时出错: {str(e)}")
                # 保留声明但标记为未接地
                claim.status = ClaimStatus.EXTRACTED
                return claim
        
        # 处理所有声明
        grounded_claims = []
        for claim in claims:
            grounded_claims.append(_ground_claim(claim))
        
        # 输出统计信息
        print(f"\n[SemanticGroundingAgent] 接地统计:")
        print(f"  处理的实体总数: {self.stats['total_entities']}")
        print(f"  概念匹配数: {self.stats['concept_matches']}")
        print(f"  实例匹配数: {self.stats['instance_matches']}")
        print(f"  新建概念数: {self.stats['new_concepts']}")
        print(f"  新建实例数: {self.stats['new_instances']}")
        print(f"  LLM 调用次数: {self.stats['llm_calls']}")
        print(f"  缓存命中数: {self.stats['cache_hits']}")
        print(f"[SemanticGroundingAgent] 完成 {len(grounded_claims)} 个声明的接地")
        
        return grounded_claims
    
    def get_stats(self) -> Dict[str, Any]:
        """获取接地统计信息。"""
        # 计算缓存命中率
        total_resolved = self.stats['total_entities']
        cache_hit_rate = self.stats['cache_hits'] / total_resolved if total_resolved > 0 else 0
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate
        }
    
    def reset_stats(self):
        """重置统计信息。"""
        self.stats = {
            'total_entities': 0,
            'concept_matches': 0,
            'instance_matches': 0,
            'new_concepts': 0,
            'new_instances': 0,
            'llm_calls': 0,
            'cache_hits': 0
        }
        print("[SemanticGroundingAgent] 统计信息已重置")
    
    def clear_cache(self):
        """清除实体解析缓存。"""
        with self._cache_lock:
            self._entity_cache.clear()
        print("[SemanticGroundingAgent] 实体缓存已清除")
