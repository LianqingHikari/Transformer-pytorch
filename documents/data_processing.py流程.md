# 1 全量数据预处理
这一步骤需要完成**分词器训练**和**文本预处理**两个步骤，两个步骤均有缓存机制，优先读取训练好的分词器和预处理好的文本。
- 分词器训练：采用BPE分词器，包含在tokenizer.py中
- 文本预处理：包含在data_processing.load_and_clean_parallel_corpus中

# 2 单样本级预处理
data_processing.WMTTranslationDataset.\_\_getitem\_\_
采用分词器对句子进行编码

# 3 批次级预处理
data_processing.collate_fn
进行填充，返回填充后的批次及其掩码