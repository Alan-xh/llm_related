# NLP 训练任务

本目录现在包含三类基于 Transformers 的可运行训练示例：

- `text_classification.py`：单标签文本分类。
- `intent_recognition.py`：意图识别，复用文本分类训练和评估代码，默认读取 `intent` 字段。
- `NER/span_ner.py`：Span-based 命名实体识别，按实体起止位置分类，支持嵌套实体。

代码面向学习和实验使用，模型权重和数据集需要自行准备。

## 文本分类

训练、验证和测试数据均使用 JSONL，每行一个样本：

```json
{"text": "这家餐厅很好吃", "label": "positive"}
{"text": "物流太慢了", "label": "negative"}
```

运行：

```bash
python -m nlp.text_classification \
  --train-file data/classification/train.jsonl \
  --valid-file data/classification/valid.jsonl \
  --test-file data/classification/test.jsonl \
  --model bert-base-chinese \
  --output outputs/text_classifier.pt
```

评估输出包括 accuracy、macro-F1、weighted-F1 和每个标签的 classification report。最佳 checkpoint 按验证集
macro-F1 保存。

## 意图识别

意图识别使用相同的数据结构，但默认字段名为 `intent`：

```json
{"text": "帮我查一下订单到哪里了", "intent": "query_logistics"}
{"text": "我想申请退款", "intent": "request_refund"}
```

运行：

```bash
python -m nlp.intent_recognition \
  --train-file data/intent/train.jsonl \
  --valid-file data/intent/valid.jsonl \
  --test-file data/intent/test.jsonl \
  --model bert-base-chinese \
  --output outputs/intent_classifier.pt
```

如果字段名不同，可以通过 `--label-field` 覆盖默认值。

## Span NER

Span NER 的实体偏移量使用 Python 字符串索引，`end` 为开区间：

```json
{
  "text": "张三在北京工作",
  "entities": [
    {"start": 0, "end": 2, "label": "PER"},
    {"start": 3, "end": 5, "label": "LOC"}
  ]
}
```

每个候选 token span 被分类为 `NONE` 或实体类型。评估时只有实体类型、字符起点和字符终点都相同才算正确，
输出严格匹配的 precision、recall、F1，并提供按实体类型统计的结果。嵌套实体可以同时存在，只要它们的边界
不同。

运行：

```bash
python -m nlp.NER.span_ner \
  --train-file data/ner/train.jsonl \
  --valid-file data/ner/valid.jsonl \
  --test-file data/ner/test.jsonl \
  --model bert-base-chinese \
  --max-span-width 10 \
  --output outputs/span_ner.pt
```

关键参数：

- `--max-length`：Transformer 输入最大 token 长度。
- `--max-span-width`：枚举的最大实体 token 宽度，应该覆盖数据中最长实体。
- `--entity-field`：实体数组字段名，默认是 `entities`。

作为库使用时，可以直接调用：

```python
from nlp.NER.span_ner import evaluate, predict, train_span_ner
from nlp.text_classification import predict as predict_class
```

注意：如果实体被 `max-length` 截断，或实体长度超过 `max-span-width`，该实体不会被映射为训练候选；
训练前应根据数据集长度分布设置这两个参数。
