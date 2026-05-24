"""
Biomedical literature extraction agent.
Supports ontology-layer and instance-layer claim extraction from paper text.
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

from schema import Paper, KnowledgeClaim, ClaimStatus
from core.llm_client import LLMClient


class ExtractionAgent:
    """Extract ontology-layer and instance-layer claims from biomedical papers."""
    
    def __init__(self, 
                 llm_client: LLMClient, 
                 config_path: Optional[str] = None,
                 examples_path: Optional[str] = None):
        """
        初始化提取代理。
        
        Args:
            llm_client: 用于声明提取的 LLM 客户端
            config_path: 配置 YAML 文件的路径
            examples_path: 示例 YAML 文件的路径（会覆盖配置）
        """
        self.llm_client = llm_client
        self.config = self._load_config(config_path)
        
        # 加载示例（优先级：参数 > 配置 > 默认）
        if examples_path:
            self.examples = self._load_examples(examples_path)
        else:
            examples_path = self.config.get('examples', {}).get('external_path')
            if examples_path and Path(examples_path).exists():
                self.examples = self._load_examples(examples_path)
            else:
                self.examples = self._get_default_examples()

        if self._looks_like_legacy_domain_examples(self.examples):
            print("[ExtractionAgent] 检测到旧域示例，回退到默认医学文献示例")
            self.examples = self._get_default_examples()
        
        # 每个线程独立持有一个 LLMClient，避免共享 httpx.Client 导致并行退化为串行
        self._thread_local = threading.local()
        self.allowed_ontology_relations = set(
            self.config.get("allowed_relations", {}).get(
                "ontology",
                [
                    "subclass_of",
                    "type",
                    "biomarker_for",
                    "risk_factor_for",
                    "treats",
                    "measured_in",
                    "part_of",
                    "associated_with",
                    "inhibits",
                    "activates",
                    "causes",
                    "predicts",
                ],
            )
        )
        self.allowed_instance_relations = set(
            self.config.get("allowed_relations", {}).get(
                "instance",
                [
                    "has_diagnosis",
                    "received_treatment",
                    "combined_with",
                    "improves",
                    "reduces",
                    "increases",
                    "associated_with",
                    "predicts",
                    "has_adverse_event",
                    "measured_value",
                    "measured_in",
                ],
            )
        )
        self.allowed_relations = self.allowed_ontology_relations | self.allowed_instance_relations
        
        # 初始化统计信息
        self.stats = {
            'total_chunks': 0,
            'total_claims': 0,
            'failed_chunks': 0,
            'ontology_claims': 0,
            'instance_claims': 0
        }
        
        max_workers = self.config.get('max_workers', 8)
        print(f"[ExtractionAgent] 初始化完成，示例数量: {len(self.examples)}")
        print(f"[ExtractionAgent] 配置: chunk_size={self.config.get('chunk_size', 6000)}, max_workers={max_workers}")

    def _infer_chunk_section_title(self, chunk: str) -> str:
        """Infer the section title encoded in a chunk, when available."""
        first_line = (chunk or "").strip().splitlines()[0].strip() if chunk.strip() else ""
        if first_line.lower().startswith("section:"):
            return first_line.split(":", 1)[1].strip()
        return ""

    def _build_extraction_context(self, paper: Paper) -> str:
        """Build a paper-level extraction context with explicit subdomain hints."""
        context_lines = [f"Title: {paper.title.strip()}"]

        subdomain = str(paper.metadata.get("subdomain", "")).strip()
        parent_domain = str(paper.metadata.get("parent_domain", "")).strip()
        if subdomain:
            context_lines.append(f"Assigned subdomain: {subdomain}")
        if parent_domain:
            context_lines.append(f"Parent domain: {parent_domain}")

        abstract = paper.get_abstract()
        if abstract:
            context_lines.append(f"Abstract: {abstract}")

        keywords = paper.get_keywords()
        if keywords:
            context_lines.append(f"Keywords: {', '.join(keywords)}")

        fallback_summary = ""
        if hasattr(paper, "_build_content_summary"):
            fallback_summary = paper._build_content_summary()
        if fallback_summary and not abstract:
            context_lines.append(f"Content summary: {fallback_summary}")

        context_lines.append(
            "Use the assigned subdomain only as a soft prior for interpreting ambiguous terms. "
            "Do not invent any fact that is not explicitly supported by the chunk."
        )
        return "\n".join(context_lines)

    def _build_claim_metadata(self, paper: Paper, chunk_idx: int, chunk: str) -> Dict[str, Any]:
        """Attach useful provenance metadata to extracted claims."""
        metadata: Dict[str, Any] = {
            "paper_title": paper.title,
            "chunk_index": chunk_idx,
        }

        subdomain = str(paper.metadata.get("subdomain", "")).strip()
        parent_domain = str(paper.metadata.get("parent_domain", "")).strip()
        section_title = self._infer_chunk_section_title(chunk)

        if subdomain:
            metadata["paper_subdomain"] = subdomain
        if parent_domain:
            metadata["paper_parent_domain"] = parent_domain
        if section_title:
            metadata["section_title"] = section_title

        return metadata
    
    def _get_llm_client(self) -> LLMClient:
        """每个线程第一次调用时创建自己独立的 LLMClient 实例。"""
        if not hasattr(self._thread_local, 'client'):
            self._thread_local.client = LLMClient(
                model_name=self.llm_client.model_name,
                base_url=self.llm_client.base_url,
                timeout=self.llm_client.timeout,
            )
        return self._thread_local.client
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """从 YAML 文件加载配置。"""
        default_config = {
            'chunk_size': 6000,
            'examples': {'use_default': True},
            'strategy': {
                'atomic_entities': True,
                'keep_original_terms': True,
                'max_claims_per_chunk': 30
            },
            'relation_normalization': {'enabled': True}
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                # 合并默认配置
                default_config.update(loaded_config)
        
        return default_config
    
    def _load_examples(self, path: str) -> List[Dict[str, Any]]:
        """从 YAML 文件加载示例。"""
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            # 处理扁平列表和分类结构
            if isinstance(data, list):
                examples = data
            elif isinstance(data, dict):
                for category, category_examples in data.items():
                    if isinstance(category_examples, list):
                        examples.extend(category_examples)
        return examples
    
    def _get_default_examples(self) -> List[Dict[str, Any]]:
        """Provide concise biomedical few-shot examples."""
        return [
            {
                "category": "Disease And Treatment",
                "text": (
                    "Hepatocellular carcinoma is the most common form of liver cancer. "
                    "The patient received nivolumab followed by lenvatinib and stereotactic body radiation therapy."
                ),
                "ontology": [
                    {
                        "subject": "hepatocellular carcinoma",
                        "relation": "subclass_of",
                        "object": "liver cancer",
                        "evidence": "Hepatocellular carcinoma is the most common form of liver cancer."
                    },
                    {
                        "subject": "nivolumab",
                        "relation": "type",
                        "object": "immune checkpoint inhibitor",
                        "evidence": "The patient received nivolumab followed by lenvatinib and stereotactic body radiation therapy."
                    },
                    {
                        "subject": "stereotactic body radiation therapy",
                        "relation": "type",
                        "object": "radiation therapy",
                        "evidence": "The patient received nivolumab followed by lenvatinib and stereotactic body radiation therapy."
                    },
                ],
                "instances": [
                    {
                        "subject": "patient",
                        "relation": "has_diagnosis",
                        "object": "hepatocellular carcinoma",
                        "evidence": "The patient received nivolumab followed by lenvatinib and stereotactic body radiation therapy."
                    },
                    {
                        "subject": "patient",
                        "relation": "received_treatment",
                        "object": "nivolumab",
                        "evidence": "The patient received nivolumab followed by lenvatinib and stereotactic body radiation therapy."
                    },
                    {
                        "subject": "nivolumab",
                        "relation": "combined_with",
                        "object": "lenvatinib",
                        "evidence": "The patient received nivolumab followed by lenvatinib and stereotactic body radiation therapy."
                    },
                ],
            },
            {
                "category": "Biomarker",
                "text": (
                    "PD-L1 expression on circulating tumor cells provided real-time insight into response to PD-1/PD-L1 inhibitors."
                ),
                "ontology": [
                    {
                        "subject": "PD-L1 expression",
                        "relation": "biomarker_for",
                        "object": "response to PD-1/PD-L1 inhibitors",
                        "evidence": "PD-L1 expression on circulating tumor cells provided real-time insight into response to PD-1/PD-L1 inhibitors."
                    },
                    {
                        "subject": "circulating tumor cells",
                        "relation": "type",
                        "object": "cancer cell population",
                        "evidence": "PD-L1 expression on circulating tumor cells provided real-time insight into response to PD-1/PD-L1 inhibitors."
                    },
                ],
                "instances": [
                    {
                        "subject": "PD-L1 expression",
                        "relation": "measured_in",
                        "object": "circulating tumor cells",
                        "evidence": "PD-L1 expression on circulating tumor cells provided real-time insight into response to PD-1/PD-L1 inhibitors."
                    }
                ],
            },
            {
                "category": "Risk Factor",
                "text": (
                    "Hepatocellular carcinoma is associated with hepatitis B virus, hepatitis C virus, and alcoholic fatty liver disease."
                ),
                "ontology": [
                    {
                        "subject": "hepatitis B virus",
                        "relation": "risk_factor_for",
                        "object": "hepatocellular carcinoma",
                        "evidence": "Hepatocellular carcinoma is associated with hepatitis B virus, hepatitis C virus, and alcoholic fatty liver disease."
                    },
                    {
                        "subject": "hepatitis C virus",
                        "relation": "risk_factor_for",
                        "object": "hepatocellular carcinoma",
                        "evidence": "Hepatocellular carcinoma is associated with hepatitis B virus, hepatitis C virus, and alcoholic fatty liver disease."
                    },
                ],
                "instances": [],
            },
        ]

    def _looks_like_legacy_domain_examples(self, examples: List[Dict[str, Any]]) -> bool:
        """Detect whether loaded examples still belong to the old ancient-text domain."""
        if not examples:
            return False

        sample_text = " ".join(str(example.get("text", "")) for example in examples[:5]).strip()
        legacy_markers = ["黄帝内经", "五脏", "六腑", "经络", "证型", "酸入肝", "汪五三"]
        return any(marker in sample_text for marker in legacy_markers)
    
    def _format_examples(self, max_examples: int = 15) -> str:
        """Format few-shot examples for biomedical extraction prompts."""
        formatted = []
        examples_to_use = self.examples[:max_examples]
        
        # 按类别分组示例
        categories = {}
        for example in examples_to_use:
            cat = example.get("category", "General")
            
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(example)
        
        # 格式化输出
        for cat, ex_list in categories.items():
            formatted.append(f"\n【{cat}】")
            for i, example in enumerate(ex_list, 1):
                formatted.append(f"示例 {i}：")
                formatted.append(f"原文：{example['text']}")
                
                # 处理本体层（支持字典格式）
                if example.get("ontology"):
                    formatted.append("  本体层提取：")
                    for claim in example["ontology"]:
                        if isinstance(claim, dict):
                            subj = claim.get("subject", "")
                            rel = claim.get("relation", "")
                            obj = claim.get("object", "")
                            evi = claim.get("evidence", "")
                            formatted.append(f"    - ({subj}, {rel}, {obj})")
                            formatted.append(f"      证据：{evi}")
                        else:
                            # 兼容旧格式字符串
                            formatted.append(f"    - {claim}")
                
                # 处理实例层
                if example.get("instances"):
                    formatted.append("  实例层提取：")
                    for claim in example["instances"]:
                        subj = claim.get("subject", "")
                        rel = claim.get("relation", "")
                        obj = claim.get("object", "")
                        evi = claim.get("evidence", "")
                        formatted.append(f"    - ({subj}, {rel}, {obj})")
                        formatted.append(f"      证据：{evi}")
                
                formatted.append("")
        
        return "\n".join(formatted)
    
    def _chunk_paper(self, paper: Paper) -> List[str]:
        """Split biomedical paper text into section-aware chunks."""
        content = paper.content.strip()
        chunk_size = self.config.get('chunk_size', 6000)
        
        if len(content) <= chunk_size:
            return [content]

        section_blocks: List[str] = []
        if "## " in content:
            section_pattern = re.compile(r"(?m)^##\s+(.+?)\n")
            matches = list(section_pattern.finditer(content))
            for idx, match in enumerate(matches):
                section_title = match.group(1).strip()
                body_start = match.end()
                body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
                body = content[body_start:body_end].strip()
                section_text = f"Section: {section_title}\n{body}" if body else f"Section: {section_title}"
                section_blocks.append(section_text.strip())
        else:
            section_blocks = [content]

        chunks: List[str] = []
        for block in section_blocks:
            if len(block) <= chunk_size:
                chunks.append(block)
                continue

            paragraphs = [part.strip() for part in re.split(r'\n\s*\n', block) if part.strip()]
            current_chunk = ""
            for para in paragraphs:
                if len(para) > chunk_size:
                    sentence_parts = [
                        part.strip() for part in re.split(r'(?<=[\.\!\?。！？])\s+', para) if part.strip()
                    ]
                    for sent in sentence_parts or [para]:
                        separator = "\n\n" if current_chunk else ""
                        candidate = f"{current_chunk}{separator}{sent}" if current_chunk else sent
                        if current_chunk and len(candidate) > chunk_size:
                            chunks.append(current_chunk.strip())
                            current_chunk = sent
                        else:
                            current_chunk = candidate
                    continue

                separator = "\n\n" if current_chunk else ""
                candidate = f"{current_chunk}{separator}{para}" if current_chunk else para
                if current_chunk and len(candidate) > chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk = candidate

            if current_chunk:
                chunks.append(current_chunk.strip())
        
        print(f"[ExtractionAgent] 将论文分割为 {len(chunks)} 个块")
        return chunks

    def _build_evidence_with_context(self, evidence: str, chunk: str, max_len: int = 1200) -> str:
        """
        将模型给出的证据句扩展为“证据句 + 上下文段落”，便于人工核查。
        """
        evidence = (evidence or "").strip()
        chunk = (chunk or "").strip()
        if not evidence:
            return ""
        if not chunk:
            return evidence

        # 先按段落查找包含证据句的段
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chunk) if p.strip()]
        context_para = ""
        for i, para in enumerate(paragraphs):
            if evidence in para or para in evidence:
                # 拼接相邻段落，提升核查可读性
                prev_para = paragraphs[i - 1] if i - 1 >= 0 else ""
                next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
                context_para = "\n\n".join([p for p in [prev_para, para, next_para] if p]).strip()
                break

        # 段落未命中，退化为基于字符窗口
        if not context_para:
            idx = chunk.find(evidence)
            if idx >= 0:
                start = max(0, idx - 260)
                end = min(len(chunk), idx + len(evidence) + 260)
                context_para = chunk[start:end].strip()
            else:
                context_para = chunk[:520].strip()

        merged = f"证据句：{evidence}\n上下文：{context_para}"
        if len(merged) > max_len:
            merged = merged[:max_len] + "..."
        return merged
    
    def _extract_claims_from_chunk(
        self,
        chunk: str,
        paper: Paper,
        chunk_idx: int,
        paper_context: str = "",
    ) -> List[KnowledgeClaim]:
        """Extract biomedical claims from a chunk."""
        
        system_prompt = """You are a biomedical literature annotation expert.

Your task is to extract two layers of claims from biomedical paper text.

1. Ontology layer
- Reusable concept-level knowledge, not tied to one specific patient or experiment.
- Use only these ontology relations:
  subclass_of, type, biomarker_for, risk_factor_for, treats, measured_in, part_of, associated_with, inhibits, activates, causes, predicts.
- Good examples:
  - hepatocellular carcinoma subclass_of liver cancer
  - nivolumab type immune checkpoint inhibitor
  - PD-L1 expression biomarker_for response to PD-1/PD-L1 inhibitors

2. Instance layer
- Study-specific, cohort-specific, patient-specific, or experiment-specific findings and events.
- Use only these instance relations:
  has_diagnosis, received_treatment, combined_with, improves, reduces, increases, associated_with, predicts, has_adverse_event, measured_value, measured_in.
- Good examples:
  - patient has_diagnosis hepatocellular carcinoma
  - nivolumab combined_with lenvatinib
  - AFP level reduced_after treatment

Rules:
- Use only information explicitly stated in the text.
- Keep entity names close to the source wording; do not invent normalization.
- Split coordinated facts into separate triples when the text clearly lists multiple entities.
- Use the paper-level subdomain as a soft prior for disambiguation only; never use it to invent unsupported triples.
- If a relation cannot be expressed with the allowed labels above, do not output the triple.
- Prefer disease, biomarker, treatment, molecular factor, outcome, and patient-level entities.
- Do not use study/reporting entities such as study, analysis, authors, we, results, findings, cohort description sentences, or entire clauses as graph nodes.
- Do not extract author metadata, funding statements, figure captions, table labels, page headers, or references.
- Do not output vague triples with section labels like Abstract, Introduction, Figure, Table.
- evidence must be a verbatim span from the chunk.
- Prefer short, controlled relation labels in English snake_case.

Return strict JSON:
{
  "ontology": [
    {"subject": "...", "relation": "...", "object": "...", "evidence": "..."}
  ],
  "instances": [
    {"subject": "...", "relation": "...", "object": "...", "evidence": "..."}
  ]
}"""

        examples = self._format_examples()
        
        user_prompt = f"""Extract ontology-layer and instance-layer biomedical triples from the paper chunk below.

Paper context:
{paper_context}

Few-shot examples:
{examples}

Chunk:
{chunk}

Output JSON:
{{
  "ontology": [{{"subject": "...", "relation": "...", "object": "...", "evidence": "..."}}],
  "instances": [{{"subject": "...", "relation": "...", "object": "...", "evidence": "..."}}]
}}"""

        try:
            print(f"[ExtractionAgent] 处理块 {chunk_idx + 1}...")
            claim_metadata = self._build_claim_metadata(paper, chunk_idx, chunk)
            
            response = self._get_llm_client().generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
            
            # 解析响应
            ontology_claims, instance_claims = self._parse_llm_response(response)
            
            # 转换为 KnowledgeClaim 对象
            claims = []
            
            # 添加本体层声明，标记类型
            for item in ontology_claims:
                try:
                    # 只处理字典格式（新格式），忽略字符串（旧格式无法获取正确证据）
                    if isinstance(item, dict):
                        subject = item.get("subject", "").strip()
                        relation = item.get("relation", "").strip()
                        obj = item.get("object", "").strip()
                        evidence_raw = item.get("evidence", "").strip()
                        
                        if subject and relation and obj and evidence_raw:
                            relation = self._normalize_relation(relation, "ontology")
                            if not relation:
                                continue
                            # 实体质量校验
                            if not self._is_valid_entity(subject, "ontology"):
                                print(f"[ExtractionAgent] 跳过无效主体实体: '{subject}'")
                                continue
                            if not self._is_valid_entity(obj, "ontology"):
                                print(f"[ExtractionAgent] 跳过无效客体实体: '{obj}'")
                                continue

                            evidence = self._build_evidence_with_context(evidence_raw, chunk)
                            
                            claim = KnowledgeClaim(
                                subject=subject,
                                relation=relation,
                                object=obj,
                                evidence=evidence,
                                source_paper_id=paper.id,
                                status=ClaimStatus.EXTRACTED,
                                metadata=dict(claim_metadata),
                            )
                            claim.claim_type = "ontology"
                            claims.append(claim)
                            self.stats['ontology_claims'] += 1
                    else:
                        # 若遇到字符串格式，打印警告并跳过
                        print(f"[ExtractionAgent] 警告：本体层声明为字符串格式，已跳过（无法获取证据原文）：{item}")
                except Exception as e:
                    print(f"[ExtractionAgent] 解析本体声明失败: {str(e)}")
                    continue
            
            # 添加实例层声明
            for item in instance_claims:
                try:
                    subject = item.get("subject", "").strip()
                    relation = item.get("relation", "").strip()
                    obj = item.get("object", "").strip()
                    evidence_raw = item.get("evidence", "").strip()
                    
                    if subject and relation and obj and evidence_raw:
                        relation = self._normalize_relation(relation, "instance")
                        if not relation:
                            continue
                        # 实体质量校验
                        if not self._is_valid_entity(subject, "instance"):
                            print(f"[ExtractionAgent] 跳过无效主体实体: '{subject}'")
                            continue
                        if not self._is_valid_entity(obj, "instance"):
                            print(f"[ExtractionAgent] 跳过无效客体实体: '{obj}'")
                            continue

                        evidence = self._build_evidence_with_context(evidence_raw, chunk)
                        claim = KnowledgeClaim(
                            subject=subject,
                            relation=relation,
                            object=obj,
                            evidence=evidence,
                            source_paper_id=paper.id,
                            status=ClaimStatus.EXTRACTED,
                            metadata=dict(claim_metadata),
                        )
                        claim.claim_type = "instance"
                        claims.append(claim)
                        self.stats['instance_claims'] += 1
                        
                except Exception as e:
                    print(f"[ExtractionAgent] 解析实例声明失败: {str(e)}")
                    continue
            
            # 更新统计信息
            self.stats['total_chunks'] += 1
            self.stats['total_claims'] += len(claims)
            
            print(f"[ExtractionAgent] 从块 {chunk_idx + 1} 提取了 {len(ontology_claims)} 个本体 + {len(instance_claims)} 个实例声明")
            return claims
            
        except Exception as e:
            print(f"[ExtractionAgent] 处理块 {chunk_idx + 1} 时出错: {str(e)}")
            self.stats['failed_chunks'] += 1
            return []
    
    def _parse_llm_response(self, response) -> tuple[List, List]:
        """
        解析和清理 LLM 响应，分离本体和实例。
        
        Returns:
            tuple: (本体列表, 实例列表)
            本体列表: 字典列表（新格式）
            实例列表: 字典列表
        """
        ontology_claims = []
        instance_claims = []
        
        if isinstance(response, dict):
            # 直接字典响应
            ontology_claims = response.get("ontology", [])
            instance_claims = response.get("instances", [])
            return ontology_claims, instance_claims
            
        if not isinstance(response, str):
            response = str(response)
        
        # 清理 markdown 并解析 JSON
        response_clean = response.strip()
        
        # 移除代码块
        code_block_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```\s*$'
        match = re.match(code_block_pattern, response_clean, re.DOTALL)
        if match:
            response_clean = match.group(1).strip()
        else:
            for prefix in ["```json", "```"]:
                if response_clean.startswith(prefix):
                    response_clean = response_clean[len(prefix):].strip()
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3].strip()
        
        try:
            parsed = json.loads(response_clean)
            
            # 处理不同的响应结构
            if isinstance(parsed, dict):
                ontology_claims = parsed.get("ontology", [])
                instance_claims = parsed.get("instances", [])
                
                # 如果没有本体/实例，检查扁平列表
                if not ontology_claims and not instance_claims:
                    for key in ["claims", "results", "data", "items"]:
                        if key in parsed and isinstance(parsed[key], list):
                            # 视为实例声明
                            instance_claims = parsed[key]
                            break
            
            elif isinstance(parsed, list):
                # 扁平列表 - 视为实例声明
                instance_claims = parsed
                
        except json.JSONDecodeError as e:
            print(f"[ExtractionAgent] JSON 解析错误: {str(e)}")
            print(f"响应预览: {response_clean[:200]}...")
        
        return ontology_claims, instance_claims
    
    def _normalize_relation(self, relation: str, claim_type: Optional[str] = None) -> str:
        relation = (relation or "").strip()
        if not relation:
            return relation

        biomedical_mapping = {
            "subclassof": "subclass_of",
            "subclass_of": "subclass_of",
            "subtype_of": "subclass_of",
            "is_a": "type",
            "type_of": "type",
            "instance_of": "type",
            "rdf_type": "type",
            "type": "type",
            "risk_factor": "risk_factor_for",
            "risk_factor_for": "risk_factor_for",
            "associated_with": "associated_with",
            "associated": "associated_with",
            "correlated_with": "associated_with",
            "biomarker_of": "biomarker_for",
            "biomarker_for": "biomarker_for",
            "measured_by": "measured_by",
            "measured_in": "measured_in",
            "treated_with": "received_treatment",
            "received_treatment": "received_treatment",
            "combination_with": "combined_with",
            "combined_with": "combined_with",
            "causes": "causes",
            "induces": "causes",
            "inhibits": "inhibits",
            "activates": "activates",
            "improves": "improves",
            "reduced": "reduced",
            "reduces": "reduces",
            "increased": "increased",
            "increases": "increases",
            "predicts": "predicts",
            "has_diagnosis": "has_diagnosis",
            "has_symptom": "has_symptom",
            "has_adverse_event": "has_adverse_event",
            "measured_value": "measured_value",
            "improved": "improves",
            "improves": "improves",
            "improvement_in": "improves",
            "reduced_after": "reduces",
            "reduction_in": "reduces",
            "reduction_of": "reduces",
            "reduces": "reduces",
            "increase_in": "increases",
            "increase_of": "increases",
            "increased": "increases",
            "had_diagnosis": "has_diagnosis",
            "diagnosed_with": "has_diagnosis",
            "treated_with": "received_treatment",
            "received": "received_treatment",
            "therapy_with": "received_treatment",
        }

        relation = re.sub(r"([a-z])([A-Z])", r"\1_\2", relation)
        ascii_relation = relation.lower().replace("-", "_").replace(" ", "_")
        ascii_relation = re.sub(r"[^a-z_]", "", ascii_relation)
        ascii_relation = re.sub(r"_+", "_", ascii_relation).strip("_")
        normalized = biomedical_mapping.get(ascii_relation, ascii_relation or relation)

        if claim_type == "ontology" and normalized not in self.allowed_ontology_relations:
            return ""
        if claim_type == "instance" and normalized not in self.allowed_instance_relations:
            return ""
        if claim_type is None and normalized not in self.allowed_relations:
            return ""

        return normalized

    def _is_valid_entity(self, entity_name: str, claim_type: str) -> bool:
        if not entity_name or not entity_name.strip():
            return False

        entity_name = entity_name.strip()
        entity_lower = entity_name.lower()

        if len(entity_name) < 2 or len(entity_name) > 120:
            return False

        artifact_terms = {
            "abstract", "introduction", "discussion", "conclusions", "conclusion",
            "results", "methods", "materials", "keywords", "figure", "fig", "table",
            "supplementary", "supporting information", "acknowledgements", "funding",
            "reference", "references", "graphical abstract", "page"
        }
        generic_meta_entities = {
            "study", "this study", "our study", "present study", "current study",
            "analysis", "result", "results", "finding", "findings", "data",
            "authors", "researchers", "paper", "article", "review", "work",
            "case", "cases", "presented case", "reported case", "literature",
        }
        if entity_lower in artifact_terms:
            return False
        if entity_lower in generic_meta_entities:
            return False
        if re.match(r'^(fig|figure|table|section|page)\b', entity_lower):
            return False
        if re.match(r'^(this|these|those|our)\s+(study|analysis|result|finding|paper|work|review)\b', entity_lower):
            return False
        if entity_name.endswith('：') or entity_name.endswith(':'):
            return False
        if len(entity_name.split()) > 15:
            return False
        if re.fullmatch(r'[\W_]+', entity_name):
            return False
        if sum(1 for ch in entity_name if ch.isalnum()) < 2:
            return False
        if self._looks_like_sentence_fragment(entity_lower):
            return False
        if claim_type == "instance" and entity_lower in {"we", "they", "patients", "subjects", "participants"}:
            return False

        return True

    def process(self, paper: Paper) -> List[KnowledgeClaim]:
        """处理论文并提取知识声明，支持本体层次。"""
        print(f"\n[ExtractionAgent] 处理论文: {paper.id} - {paper.title}")
        
        chunks = self._chunk_paper(paper)
        paper_context = self._build_extraction_context(paper)
        max_workers = self.config.get('max_workers', 8)
        print(f"[ExtractionAgent] 使用 {max_workers} 个并行工作线程处理 {len(chunks)} 个块")

        all_claims = []

        def extract_one(args):
            idx, chunk = args
            return self._extract_claims_from_chunk(chunk, paper, idx, paper_context)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_one, (i, c)): i for i, c in enumerate(chunks)}
            for future in as_completed(futures):
                try:
                    all_claims.extend(future.result())
                except Exception as e:
                    print(f"[ExtractionAgent] 块处理出错: {e}")

        # 去重
        all_claims = self._deduplicate_claims(all_claims)
        all_claims = self._filter_claims(all_claims)
        all_claims = self._cap_claims_per_chunk(all_claims)
        
        # 统计本体和实例数量
        ontology_count = sum(1 for c in all_claims if hasattr(c, 'claim_type') and c.claim_type == 'ontology')
        instance_count = len(all_claims) - ontology_count
        
        print(f"[ExtractionAgent] 总共提取了 {len(all_claims)} 个唯一声明: {ontology_count} 本体 + {instance_count} 实例")
        return all_claims
    
    def _filter_claims(self, claims: List[KnowledgeClaim]) -> List[KnowledgeClaim]:
        """Post-filter obvious extraction noise for biomedical claims."""
        relation_blacklist = {"related", "relation", "relevant", "link", "mentioned"}
        artifact_pattern = re.compile(
            r"^(abstract|introduction|discussion|conclusions?|results|methods?|keywords?|figure|fig|table|supplementary|page)\b",
            re.IGNORECASE,
        )
        meta_evidence_pattern = re.compile(
            r"\b(this study|our study|we (investigated|evaluated|analyzed|examined|show|showed|demonstrated|found|propose|present)|"
            r"the authors|retrospective study|prospective study|randomized study|in this study)\b",
            re.IGNORECASE,
        )
        filtered, dropped = [], []

        for claim in claims:
            subj = claim.subject.strip()
            obj = claim.object.strip()
            rel = claim.relation.strip()
            ev = (claim.evidence or "").strip()
            section_title = str((claim.metadata or {}).get("section_title", "")).strip().lower()

            if subj == obj:
                dropped.append(f"自环: {subj} --{rel}-->")
                continue
            if not ev or len(ev) < 10:
                dropped.append(f"证据过短: {subj} --{rel}--> {obj}")
                continue
            if artifact_pattern.match(subj) or artifact_pattern.match(obj):
                dropped.append(f"结构词实体: {subj} --{rel}--> {obj}")
                continue
            if rel not in self.allowed_relations:
                dropped.append(f"关系不在受控集合内: {rel}")
                continue
            if len(subj.split()) > 15 or len(obj.split()) > 15:
                dropped.append(f"实体疑似整句: {subj[:30]} / {obj[:30]}")
                continue
            if len(rel) > 40 or len(rel.split()) > 5 or rel.lower() in relation_blacklist:
                dropped.append(f"关系异常: {rel}")
                continue
            if re.fullmatch(r'[\d\W_]+', subj) or re.fullmatch(r'[\d\W_]+', obj):
                dropped.append(f"纯符号实体: {subj} / {obj}")
                continue
            if self._looks_like_meta_entity(subj) or self._looks_like_meta_entity(obj):
                dropped.append(f"元叙事实体: {subj} / {obj}")
                continue
            if meta_evidence_pattern.search(ev):
                dropped.append(f"元叙事证据: {subj} --{rel}--> {obj}")
                continue
            if self._should_drop_by_section(claim.claim_type, rel, section_title):
                dropped.append(f"section 约束过滤: {section_title or 'unknown'} -> {subj} --{rel}--> {obj}")
                continue

            filtered.append(claim)

        if dropped:
            print(f"[ExtractionAgent] _filter_claims 过滤掉 {len(dropped)} 条噪音:")
            for msg in dropped[:10]:
                print(f"  - {msg}")
            if len(dropped) > 10:
                print(f"  ... 共 {len(dropped)} 条")

        return filtered

    def _deduplicate_claims(self, claims: List[KnowledgeClaim]) -> List[KnowledgeClaim]:
        """基于（主语，关系，宾语）移除重复声明。"""
        seen = set()
        unique = []
        
        for claim in claims:
            # 归一化实体
            subject = self._normalize_entity(claim.subject)
            obj = self._normalize_entity(claim.object)
            
            key = (subject, claim.relation, obj)
            if key not in seen:
                seen.add(key)
                unique.append(claim)
        
        return unique

    def _cap_claims_per_chunk(self, claims: List[KnowledgeClaim]) -> List[KnowledgeClaim]:
        """Keep the highest-value claims per chunk to reduce over-extraction."""
        max_claims = int(self.config.get("strategy", {}).get("max_claims_per_chunk", 15))
        if max_claims <= 0:
            return claims

        grouped: Dict[Any, List[KnowledgeClaim]] = {}
        for claim in claims:
            chunk_idx = (claim.metadata or {}).get("chunk_index", -1)
            grouped.setdefault(chunk_idx, []).append(claim)

        capped: List[KnowledgeClaim] = []
        trimmed = 0
        for chunk_idx in sorted(grouped.keys()):
            chunk_claims = grouped[chunk_idx]
            ranked = sorted(chunk_claims, key=self._claim_priority, reverse=True)
            capped.extend(ranked[:max_claims])
            trimmed += max(0, len(ranked) - max_claims)

        if trimmed:
            print(f"[ExtractionAgent] _cap_claims_per_chunk 额外裁剪 {trimmed} 条低优先级声明")

        return capped

    def _claim_priority(self, claim: KnowledgeClaim) -> int:
        section_title = str((claim.metadata or {}).get("section_title", "")).strip().lower()
        relation = claim.relation.strip()
        score = 0

        if claim.claim_type == "ontology":
            score += 6
        else:
            score += 3

        high_value_relations = {
            "subclass_of", "type", "biomarker_for", "risk_factor_for", "treats",
            "has_diagnosis", "received_treatment", "predicts", "has_adverse_event",
        }
        medium_value_relations = {"associated_with", "inhibits", "activates", "causes", "combined_with", "measured_in"}
        if relation in high_value_relations:
            score += 4
        elif relation in medium_value_relations:
            score += 2

        if section_title.startswith(("results", "discussion", "conclusion", "case presentation", "case report", "abstract")):
            score += 2
        if section_title.startswith(("introduction", "background")):
            score -= 2
        if any(token in section_title for token in ("methods", "experimental", "materials")):
            score -= 3

        if len(claim.subject.split()) <= 4:
            score += 1
        if len(claim.object.split()) <= 5:
            score += 1

        return score
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity names conservatively for de-duplication only."""
        entity = (entity or "").strip()
        if not entity:
            return entity

        entity = re.sub(r"\s+", " ", entity)
        entity = re.sub(r"\s*([,;:()\\[\\]/-])\s*", r"\1", entity)
        entity = entity.strip(" .;:,")
        return entity.lower()

    def _looks_like_sentence_fragment(self, entity_lower: str) -> bool:
        words = entity_lower.split()
        if len(words) <= 5:
            return False

        meta_verbs = {
            "demonstrated", "showed", "show", "found", "investigated", "evaluated",
            "analyzed", "examined", "included", "presented", "received", "used",
            "reported", "compared", "suggested", "indicated",
        }
        if any(word in meta_verbs for word in words):
            return True

        noisy_prefixes = (
            "retrospective ", "prospective ", "randomized ", "multicenter ", "single center ",
            "patient with ", "patients with ", "patient in ", "patients in ", "case of ",
        )
        return entity_lower.startswith(noisy_prefixes)

    def _looks_like_meta_entity(self, entity_name: str) -> bool:
        entity_lower = entity_name.lower().strip()
        if entity_lower in {
            "study", "this study", "our study", "analysis", "findings", "results",
            "data", "authors", "we", "review", "paper", "article", "patient cohort",
        }:
            return True
        return bool(re.match(r"^(this|our)\s+(study|analysis|work|paper|review)\b", entity_lower))

    def _should_drop_by_section(self, claim_type: Optional[str], relation: str, section_title: str) -> bool:
        if not section_title:
            return False

        if section_title.startswith(("introduction", "background")) and claim_type == "instance":
            return True

        if any(token in section_title for token in ("methods", "experimental", "materials")):
            allowed_method_relations = {"type", "subclass_of", "part_of", "measured_in"}
            return relation not in allowed_method_relations

        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取提取统计信息。"""
        return self.stats
