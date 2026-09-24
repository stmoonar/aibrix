"""
数据生成器模块

根据指定的token长度和tokenizer地址生成多样化的prompt
支持严格的token长度控制和高度的随机性
"""

import os
import random
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile

try:
    from modelscope import snapshot_download
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError(f"需要安装依赖: pip install modelscope transformers. 错误: {e}")

from .config_manager import ConfigManager


class DataGenerator:
    """数据生成器 - 生成符合指定token长度的prompt"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化数据生成器
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.tokenizers = {}  # 缓存已加载的tokenizer
        self.logger = self._setup_logger()
        
        # 初始化内容库
        self._init_content_templates()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("DataGenerator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_content_templates(self):
        """初始化内容模板库"""
        
        # 对话开始语句
        self.conversation_starters = [
            "请帮我详细分析",
            "能否深入解释",
            "我想全面了解",
            "请具体描述",
            "能否提供建议关于",
            "请帮助我理解",
            "请比较分析",
            "如何有效改进",
            "请评估并建议",
            "制定详细方案关于"
        ]
        
        # 技术领域主题
        self.tech_domains = [
            "机器学习模型优化策略",
            "分布式系统架构设计",
            "数据库查询性能调优",
            "云原生应用部署方案",
            "人工智能算法实现",
            "区块链共识机制",
            "前端性能优化技术",
            "微服务API网关设计",
            "容器编排管理策略",
            "网络安全防护体系",
            "大数据实时处理架构",
            "自动化运维流程设计"
        ]
        
        # 编程实践场景
        self.programming_scenarios = [
            "Python深度学习框架",
            "JavaScript异步编程",
            "Java企业级应用开发",
            "Go并发程序设计",
            "Rust内存安全编程",
            "C++高性能计算",
            "SQL复杂查询优化",
            "Shell系统管理脚本",
            "Docker多阶段构建",
            "Kubernetes资源管理"
        ]
        
        # 业务应用领域
        self.business_contexts = [
            "电商推荐系统算法",
            "供应链智能优化",
            "金融风控模型设计",
            "智能客服系统",
            "数据驱动营销策略",
            "用户行为分析平台",
            "实时监控告警系统",
            "多租户SaaS架构",
            "移动应用性能优化",
            "跨平台开发框架"
        ]
        
        # 扩展连接词
        self.connective_phrases = [
            "具体而言",
            "进一步说明",
            "另一方面",
            "需要特别注意",
            "从技术角度看",
            "综合考虑各因素",
            "基于最佳实践",
            "深入分析后发现",
            "关键在于理解",
            "核心问题是"
        ]
        
        # 详细要求描述
        self.detailed_requirements = [
            "包含完整的技术架构和实现细节",
            "需要考虑可扩展性和高可用性要求",
            "涉及复杂的算法设计和性能优化",
            "要求深入的理论基础和实践经验",
            "需要综合运用多种先进技术",
            "包含详细的测试策略和质量保证",
            "涵盖从设计到部署的完整流程",
            "需要结合具体业务场景和用户需求"
        ]
    
    def load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """
        加载指定模型的tokenizer，优先从本地缓存加载
        
        Args:
            model_name: 模型名称（需要在配置文件中定义）
            
        Returns:
            AutoTokenizer实例
        """
        if model_name in self.tokenizers:
            return self.tokenizers[model_name]
        
        # 从配置获取模型URL
        model_urls = self.config_manager.get_model_urls()
        if model_name not in model_urls:
            available_models = list(model_urls.keys())
            raise ValueError(f"模型 '{model_name}' 不在配置中。可用模型: {available_models}")
        
        model_url = model_urls[model_name]
        # 使用 model_url 的最后一部分作为缓存目录名
        cache_name = model_url.rstrip('/').split('/')[-1]
        cache_dir = os.path.join(tempfile.gettempdir(), "modelscope_cache", cache_name)
        
        # 指定只下载tokenizer相关文件
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json", 
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "config.json"  # tokenizer可能需要模型配置
        ]
        
        try:
            # 首先尝试从本地缓存加载
            model_path = os.path.join(cache_dir, model_url.replace('/', os.sep))
            self.logger.info(f"尝试从本地缓存加载tokenizer: {model_path}")
            if os.path.exists(model_path):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        use_fast=True,
                        local_files_only=True
                    )
                    self.tokenizers[model_name] = tokenizer
                    self.logger.info(f"✅ 成功从本地缓存加载tokenizer: {model_name}")
                    return tokenizer
                except Exception as e:
                    self.logger.warning(f"从本地缓存加载失败: {e}")
            
            # 本地加载失败，从modelscope下载
            self.logger.info(f"从modelscope下载tokenizer: {model_url}")
            model_path = snapshot_download(
                model_url, 
                cache_dir=cache_dir,
                allow_file_pattern=tokenizer_files
            )
            
            # 加载下载的tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=True
            )
            
            self.tokenizers[model_name] = tokenizer
            self.logger.info(f"✅ 成功下载并加载tokenizer: {model_name}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"❌ 加载tokenizer失败 {model_name}: {e}")
            raise RuntimeError(f"无法加载tokenizer {model_name}: {e}")
    
    def _create_base_prompt(self, content_category: str) -> str:
        """
        创建基础prompt内容
        
        Args:
            content_category: 内容类别
            
        Returns:
            基础prompt字符串
        """
        if content_category == "technical":
            starter = random.choice(self.conversation_starters)
            domain = random.choice(self.tech_domains)
            return f"{starter}{domain}的实现方案。"
            
        elif content_category == "programming":
            scenario = random.choice(self.programming_scenarios)
            requirement = random.choice(self.detailed_requirements)
            return f"请设计并实现{scenario}项目，{requirement}。"
            
        elif content_category == "business":
            context = random.choice(self.business_contexts)
            detail = random.choice(self.detailed_requirements)
            return f"需要构建{context}，{detail}。"
            
        elif content_category == "analysis":
            topic = random.choice(self.tech_domains + self.business_contexts)
            return f"深入分析{topic}的技术方案，{random.choice(self.detailed_requirements)}。"
            
        else:  # mixed - 随机选择类型
            categories = ["technical", "programming", "business", "analysis"]
            return self._create_base_prompt(random.choice(categories))
    
    def _expand_prompt_content(self, base_prompt: str, target_tokens: int, current_tokens: int) -> str:
        """
        扩展prompt内容以达到目标token数量
        
        Args:
            base_prompt: 基础prompt
            target_tokens: 目标token数量
            current_tokens: 当前token数量
            
        Returns:
            扩展后的prompt
        """
        expanded_prompt = base_prompt
        tokens_needed = target_tokens - current_tokens
        
        expansion_strategies = [
            "add_constraints",
            "add_examples",
            "add_technical_details", 
            "add_quality_requirements",
            "add_implementation_steps"
        ]
        
        while tokens_needed > 15:  # 保留缓冲空间
            strategy = random.choice(expansion_strategies)
            
            if strategy == "add_constraints":
                constraints = [
                    "在有限资源条件下实现最优性能",
                    "确保系统的高可用性和容错能力",
                    "兼容现有技术栈和业务流程",
                    "满足严格的安全性和合规要求",
                    "支持大规模并发和数据处理"
                ]
                addition = f" {random.choice(self.connective_phrases)}，{random.choice(constraints)}。"
                
            elif strategy == "add_examples":
                examples = [
                    "参考业界领先的成功实践案例",
                    "借鉴开源项目的优秀设计模式",
                    "结合实际生产环境的应用经验",
                    "采用经过验证的技术解决方案",
                    "整合多个成熟框架的核心优势"
                ]
                addition = f" 建议{random.choice(examples)}，确保方案的可行性和稳定性。"
                
            elif strategy == "add_technical_details":
                details = [
                    "包含详细的系统架构图和组件交互关系",
                    "提供核心算法的伪代码和实现逻辑",
                    "说明数据流转和状态管理的设计方案",
                    "描述接口定义和通信协议的选择",
                    "阐述存储方案和缓存策略的设计"
                ]
                addition = f" {random.choice(details)}。"
                
            elif strategy == "add_quality_requirements":
                quality_reqs = [
                    "确保代码质量通过严格的测试覆盖和代码审查",
                    "建立完善的监控体系和告警机制",
                    "制定详细的性能基准和优化目标",
                    "设计全面的错误处理和恢复策略",
                    "建立持续集成和自动化部署流程"
                ]
                addition = f" {random.choice(self.connective_phrases)}，{random.choice(quality_reqs)}。"
                
            else:  # add_implementation_steps
                steps = [
                    "制定分阶段的开发计划和里程碑",
                    "设计详细的技术评审和验收标准",
                    "建立团队协作和沟通机制",
                    "规划资源分配和风险控制措施",
                    "制定上线部署和维护运营方案"
                ]
                addition = f" 实施过程中需要{random.choice(steps)}。"
            
            expanded_prompt += addition
            # 估算添加的token数量（粗略估计）
            tokens_needed -= len(addition) * 0.6
            
            if tokens_needed <= 15:
                break
        
        return expanded_prompt
    
    def _precise_truncate(self, content: str, tokenizer: AutoTokenizer, max_tokens: int) -> str:
        """
        精确截断内容到指定token数量
        
        Args:
            content: 待截断的内容
            tokenizer: tokenizer实例
            max_tokens: 最大token数量
            
        Returns:
            截断后的内容
        """
        tokens = tokenizer.encode(content, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return content
        
        # 截断到目标长度
        truncated_tokens = tokens[:max_tokens]
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        # 优化截断点，避免在词语中间截断
        if truncated_text and not truncated_text.endswith(('。', '！', '？', '；', '\n', ' ')):
            # 向前查找合适的截断点
            for i in range(len(truncated_text) - 1, max(0, len(truncated_text) - 20), -1):
                if truncated_text[i] in ['。', '！', '？', '；', '\n', ' ', '，']:
                    truncated_text = truncated_text[:i + 1]
                    break
        
        return truncated_text.strip()
    
    def generate_prompt(self, 
                       target_token_length: int,
                       model_name: str,
                       content_category: str = "mixed",
                       randomness: float = 0.9) -> Dict[str, Any]:
        """
        生成指定token长度的prompt
        
        Args:
            target_token_length: 目标token长度
            model_name: 模型名称
            content_category: 内容类别 ("technical", "programming", "business", "analysis", "mixed")
            randomness: 随机性控制 (0.0-1.0)
            
        Returns:
            包含prompt和元数据的字典
        """
        # 加载对应的tokenizer
        tokenizer = self.load_tokenizer(model_name)
        
        # 应用随机性设置
        if randomness > 0.5:
            random.seed(None)  # 使用系统时间作为种子
        
        # 生成基础内容
        base_content = self._create_base_prompt(content_category)
        
        # 计算基础内容的token数量
        base_tokens = tokenizer.encode(base_content, add_special_tokens=False)
        base_token_count = len(base_tokens)
        
        # 根据目标长度调整内容
        if base_token_count < target_token_length:
            # 需要扩展
            expanded_content = self._expand_prompt_content(
                base_content, target_token_length, base_token_count
            )
            
            # 检查扩展后的长度
            expanded_tokens = tokenizer.encode(expanded_content, add_special_tokens=False)
            if len(expanded_tokens) > target_token_length:
                # 如果超长，精确截断
                final_content = self._precise_truncate(expanded_content, tokenizer, target_token_length)
            else:
                final_content = expanded_content
                
        elif base_token_count > target_token_length:
            # 需要截断
            final_content = self._precise_truncate(base_content, tokenizer, target_token_length)
        else:
            # 长度正好
            final_content = base_content
        
        # 最终验证token长度
        final_tokens = tokenizer.encode(final_content, add_special_tokens=False)
        actual_token_count = len(final_tokens)
        
        # 计算准确率
        accuracy = actual_token_count / target_token_length if target_token_length > 0 else 0
        
        return {
            "prompt": final_content,
            "target_token_length": target_token_length,
            "actual_token_length": actual_token_count,
            "model_name": model_name,
            "content_category": content_category,
            "randomness": randomness,
            "length_accuracy": accuracy,
            "token_efficiency": min(accuracy, 2.0 - accuracy)  # 越接近1.0越好
        }
    
    def generate_batch(self,
                      batch_size: int,
                      token_length_range: Tuple[int, int],
                      model_name: str,
                      content_categories: Optional[List[str]] = None,
                      randomness: float = 0.9) -> List[Dict[str, Any]]:
        """
        批量生成prompts
        
        Args:
            batch_size: 批次大小
            token_length_range: token长度范围 (最小值, 最大值)
            model_name: 模型名称
            content_categories: 内容类别列表
            randomness: 随机性控制
            
        Returns:
            prompt数据列表
        """
        if content_categories is None:
            content_categories = ["technical", "programming", "business", "analysis", "mixed"]
        
        prompts = []
        min_tokens, max_tokens = token_length_range
        
        self.logger.info(f"开始批量生成 {batch_size} 个prompts，token范围: {min_tokens}-{max_tokens}")
        
        for i in range(batch_size):
            # 随机选择参数
            target_length = random.randint(min_tokens, max_tokens)
            category = random.choice(content_categories)
            
            try:
                # 生成单个prompt
                prompt_data = self.generate_prompt(
                    target_token_length=target_length,
                    model_name=model_name,
                    content_category=category,
                    randomness=randomness
                )
                
                prompt_data["batch_index"] = i
                prompts.append(prompt_data)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"已生成 {i + 1}/{batch_size} 个prompts")
                    
            except Exception as e:
                self.logger.error(f"生成第 {i+1} 个prompt时出错: {e}")
                continue
        
        self.logger.info(f"✅ 批量生成完成，成功生成 {len(prompts)} 个prompts")
        return prompts
    
    def save_to_file(self, prompts: List[Dict[str, Any]], file_path: str):
        """
        将prompts保存到文件
        
        Args:
            prompts: prompt数据列表
            file_path: 输出文件路径
        """
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
        
        self.logger.info(f"💾 已保存 {len(prompts)} 个prompts到: {file_path}")
    
    def get_statistics(self, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取prompts的统计信息
        
        Args:
            prompts: prompt数据列表
            
        Returns:
            统计信息字典
        """
        if not prompts:
            return {"error": "没有数据"}
        
        actual_lengths = [p['actual_token_length'] for p in prompts]
        target_lengths = [p['target_token_length'] for p in prompts]
        accuracies = [p['length_accuracy'] for p in prompts]
        efficiencies = [p['token_efficiency'] for p in prompts]
        
        return {
            "总数": len(prompts),
            "Token长度统计": {
                "实际长度": {
                    "最小值": min(actual_lengths),
                    "最大值": max(actual_lengths),
                    "平均值": round(sum(actual_lengths) / len(actual_lengths), 2)
                },
                "目标长度": {
                    "最小值": min(target_lengths),
                    "最大值": max(target_lengths),
                    "平均值": round(sum(target_lengths) / len(target_lengths), 2)
                }
            },
            "质量指标": {
                "长度准确率": {
                    "最小值": round(min(accuracies), 3),
                    "最大值": round(max(accuracies), 3),
                    "平均值": round(sum(accuracies) / len(accuracies), 3)
                },
                "Token效率": {
                    "最小值": round(min(efficiencies), 3),
                    "最大值": round(max(efficiencies), 3),
                    "平均值": round(sum(efficiencies) / len(efficiencies), 3)
                }
            },
            "内容分布": {
                category: len([p for p in prompts if p['content_category'] == category])
                for category in set(p['content_category'] for p in prompts)
            }
        }


def create_data_generator(config_path: str = "config/default.yaml") -> DataGenerator:
    """
    创建数据生成器实例的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DataGenerator实例
    """
    config_manager = ConfigManager(config_path)
    config_manager.load_config()
    return DataGenerator(config_manager) 