# Hugging Face Llama 3.2 3B Instruct

## Download → Load Locally → FastAPI → dtype → bfloat16 → Accelerate

এই note-এ আমরা `meta-llama/Llama-3.2-3B-Instruct` model কীভাবে:

1. Hugging Face থেকে download করব
2. Local machine-এ রাখব
3. Python দিয়ে load করব
4. FastAPI route থেকে ব্যবহার করব
5. `dtype` কী এবং কেন দরকার বুঝব
6. `float32`, `float16`, `bfloat16` বুঝব
7. `device_map="auto"` বুঝব
8. `accelerate` কেন install করতে হয় বুঝব
9. শেষে পুরো architecture পরিষ্কার করব

---

# 1. আমরা কী বানাতে যাচ্ছি?

আমাদের final application হবে:

```text
                    Client
                      │
                      │ POST /chat
                      ▼
                 FastAPI
                      │
                      ▼
               generate_response()
                      │
                      ▼
             Llama 3.2 3B
                      │
                      ▼
                 AI Response
                      │
                      ▼
                   JSON
```

Example request:

```http
POST /chat
```

```json
{
  "message": "What is an API?"
}
```

Response:

```json
{
  "response": "An API is a way for..."
}
```

---

# 2. Model কী?

আমরা ব্যবহার করব:

```text
meta-llama/Llama-3.2-3B-Instruct
```

এখানে:

```text
meta-llama
```

হলো model publisher/organization.

আর:

```text
Llama-3.2-3B-Instruct
```

হলো specific model.

`3B` বলতে approximately 3 billion parameters বোঝায়।

`Instruct` version-এর উদ্দেশ্য হলো user instruction follow করে response generate করা।

অর্থাৎ এটি সাধারণ chatbot/assistant-type কাজের জন্য উপযুক্ত।

---

# 3. Project structure

আমাদের project:

```text
llama-api/
│
├── app/
│   ├── main.py
│   └── model.py
│
├── models/
│   └── llama-3.2-3b/
│
├── download_model.py
│
├── pyproject.toml
└── uv.lock
```

এখানে:

### `download_model.py`

শুধু model download করার জন্য।

### `models/llama-3.2-3b/`

এখানে downloaded model files থাকবে।

### `app/model.py`

এখানে model load হবে এবং AI inference logic থাকবে।

### `app/main.py`

এখানে FastAPI application এবং API routes থাকবে।

---

# 4. Project তৈরি করা

PowerShell:

```powershell
mkdir llama-api
cd llama-api
```

তারপর:

```powershell
uv init
```

Directories:

```powershell
mkdir app
mkdir models
```

---

# 5. Dependencies install করা

আমাদের লাগবে:

```text
torch
transformers
accelerate
huggingface-hub
fastapi
uvicorn
```

Install:

```powershell
uv add torch transformers accelerate huggingface-hub fastapi uvicorn
```

### এগুলোর কাজ

| Package           | কাজ                                                                |
| ----------------- | ------------------------------------------------------------------ |
| `torch`           | Neural network/model চালায়                                         |
| `transformers`    | Hugging Face model load/run করে                                    |
| `huggingface-hub` | Hugging Face থেকে model download করে                               |
| `accelerate`      | Large model-এর device placement/offloading manage করতে সাহায্য করে |
| `fastapi`         | API তৈরি করে                                                       |
| `uvicorn`         | FastAPI server চালায়                                               |

---

# 6. Hugging Face authentication

Llama model download করার আগে model-এর access requirements check করতে হবে।

Hugging Face account দিয়ে login:

```powershell
uv run hf auth login
```

তারপর Hugging Face access token দিতে হবে।

Check:

```powershell
uv run hf auth whoami
```

যদি username দেখায়, authentication successful।

---

# 7. Model download

`download_model.py` তৈরি করো:

```python
from huggingface_hub import snapshot_download


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_PATH = "./models/llama-3.2-3b"


snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_PATH,
)


print(f"Model downloaded to: {MODEL_PATH}")
```

Run:

```powershell
uv run python download_model.py
```

এখন model local machine-এ download হবে।

Structure হবে roughly:

```text
llama-api/
│
├── models/
│   └── llama-3.2-3b/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── generation_config.json
│       ├── *.safetensors
│       └── ...
```

### Important

এই files manually edit করার দরকার নেই।

এগুলো model-এর configuration, tokenizer এবং weights।

---

# 8. Model files-এর মধ্যে সবচেয়ে important কী?

দুইটা concept মনে রাখো:

```text
Tokenizer
+
Model weights
```

### Tokenizer

Human-readable text-কে model-এর বোঝার মতো tokens-এ convert করে।

```text
"What is Python?"
        ↓
    Tokenizer
        ↓
   Token IDs
```

### Model weights

এগুলোই মূল neural network-এর learned parameters।

যেমন:

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
```

এই `.safetensors` files-এর মধ্যে model-এর weights থাকে।

---

# 9. এখন model load করব

`app/model.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "./models/llama-3.2-3b"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
)
```

এখন এই দুইটা line খুব important:

```python
dtype="auto"
```

এবং:

```python
device_map="auto"
```

এগুলো বুঝতে হলে আগে `dtype` বুঝতে হবে।

---

# 10. `dtype` কী?

`dtype` = **data type**

একটা neural network-এর ভিতরে billions of numbers থাকে।

উদাহরণ:

```text
0.1234
-0.8721
1.2456
0.00321
...
```

এই numbers computer memory-তে কোনো না কোনো format-এ store করতে হয়।

`dtype` বলে দেয়:

> এই numbers কী ধরনের numerical representation ব্যবহার করবে?

Common types:

```python
torch.float32
torch.float16
torch.bfloat16
```

---

# 11. `float32`

সবচেয়ে সাধারণ traditional floating-point format:

```python
dtype=torch.float32
```

একটা value:

```text
32 bits = 4 bytes
```

তাই 3 billion parameters-এর raw weight storage roughly:

```text
3,000,000,000 × 4 bytes
≈ 12 GB
```

এটা শুধু একটা simplified weight calculation।

Actual runtime memory আরও বেশি হতে পারে।

---

# 12. `float16`

আরেকটি format:

```python
dtype=torch.float16
```

এখানে:

```text
16 bits = 2 bytes
```

তাই একই 3B parameters-এর raw weights roughly:

```text
3B × 2 bytes
≈ 6 GB
```

অর্থাৎ FP32-এর তুলনায় প্রায় অর্ধেক weight memory।

---

# 13. `bfloat16`

এখন আসি তোমার মূল প্রশ্নে।

অনেকে লেখে:

```python
dtype=torch.bfloat16
```

`bfloat16` = **Brain Floating Point 16**

এটিও:

```text
16 bits = 2 bytes
```

তাই BF16 এবং FP16 দুটোই প্রায় একই amount of memory ব্যবহার করে model weights-এর জন্য।

কিন্তু তাদের numerical representation আলাদা।

---

# 14. FP32 vs FP16 vs BF16

একটা সহজ comparison:

| dtype      |   Size | Main idea                       |
| ---------- | -----: | ------------------------------- |
| `float32`  | 32-bit | বেশি precision, বেশি memory     |
| `float16`  | 16-bit | কম memory, ভালো GPU performance |
| `bfloat16` | 16-bit | কম memory, বড় numerical range   |

সহজভাবে:

```text
FP32
│
├── বেশি memory
└── বেশি precision
```

```text
FP16
│
├── কম memory
└── smaller numerical range
```

```text
BF16
│
├── কম memory
└── much larger numerical range than FP16
```

---

# 15. তাহলে BF16 কেন ব্যবহার করা হয়?

BF16-এর সবচেয়ে বড় সুবিধা:

```text
FP16-এর মতো 16-bit memory usage
+
FP32-এর কাছাকাছি বড় exponent range
```

অর্থাৎ:

```text
BF16
≈ 2 bytes/value
```

কিন্তু numerical range FP16-এর চেয়ে অনেক বড়।

এ কারণে modern deep-learning workloads-এ BF16 খুব জনপ্রিয়।

বিশেষ করে modern GPUs-তে BF16 support থাকলে LLM inference/training-এর জন্য এটি ভালো choice হতে পারে।

---

# 16. `dtype=torch.bfloat16` আসলে কী বলে?

যদি লিখি:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
)
```

এর অর্থ:

> Model-টি BF16 dtype ব্যবহার করে load করো।

অর্থাৎ আমরা model-কে বলছি:

```text
"Model-এর numerical weights-এর জন্য bfloat16 ব্যবহার করো।"
```

এটা model-এর intelligence কমিয়ে দেয়—এভাবে ভাবা ঠিক নয়।

এটা মূলত numerical representation এবং memory/performance-এর বিষয়।

---

# 17. তাহলে `dtype="auto"` কী?

আমরা লিখেছিলাম:

```python
dtype="auto"
```

এর অর্থ:

> Model-এর stored/configured dtype দেখে appropriate dtype ব্যবহার করো।

এটা convenient কারণ আমাদের manually বলতে হচ্ছে না:

```python
torch.float32
```

বা:

```python
torch.float16
```

বা:

```python
torch.bfloat16
```

---

# 18. `dtype="auto"` বনাম `dtype=torch.bfloat16`

### Option 1

```python
dtype="auto"
```

মানে:

```text
"Model-এর dtype অনুযায়ী automatically handle করো."
```

### Option 2

```python
dtype=torch.bfloat16
```

মানে:

```text
"আমি explicitly BF16 ব্যবহার করতে চাই."
```

---

# 19. কোনটা ব্যবহার করা উচিত?

Learning এবং initial setup-এর সময়:

```python
dtype="auto"
```

ব্যবহার করা সহজ।

যখন তুমি জানো:

```text
আমার GPU BF16 support করে
```

তখন explicit:

```python
dtype=torch.bfloat16
```

ব্যবহার করতে পারো।

অন্যের code দেখে blindly:

```python
dtype=torch.bfloat16
```

copy করা উচিত নয়।

Hardware support গুরুত্বপূর্ণ।

---

# 20. `torch_dtype` কী?

তুমি হয়তো দেখবে:

```python
torch_dtype=torch.bfloat16
```

এবং নতুন code-এ:

```python
dtype=torch.bfloat16
```

দুটো দেখে মনে হতে পারে এগুলো আলাদা।

Conceptually এখানে মূল বিষয় একই।

`torch_dtype` হলো older/legacy argument name।

Current Transformers code/documentation-এ:

```python
dtype=
```

ব্যবহার করা preferred।

তাই নতুন code লিখলে:

```python
dtype=torch.bfloat16
```

ব্যবহার করা ভালো।

কিন্তু পুরোনো tutorial-এ:

```python
torch_dtype=torch.bfloat16
```

দেখলে ভয় পাওয়ার কিছু নেই।

---

# 21. এবার `device_map="auto"`

এটা `dtype` থেকে completely different concept।

মনে রাখো:

```text
dtype
↓
HOW?

device_map
↓
WHERE?
```

অর্থাৎ:

```python
dtype=torch.bfloat16
```

বলছে:

> Model-এর numbers কী format-এ থাকবে?

আর:

```python
device_map="auto"
```

বলছে:

> Model-এর বিভিন্ন অংশ কোন device-এ থাকবে?

---

# 22. Device বলতে কী বোঝায়?

সাধারণত:

```text
CPU
GPU
```

যেমন:

```text
CPU RAM
GPU VRAM
```

একটা বড় model পুরো GPU memory-তে fit নাও করতে পারে।

উদাহরণ:

```text
GPU VRAM = 6 GB

Model weights = 8 GB
```

তাহলে পুরো model GPU-তে fit করবে না।

---

# 23. `device_map="auto"` কী করে?

যখন লিখি:

```python
device_map="auto"
```

Transformers/Accelerate model-এর layers কোথায় রাখা সম্ভব সেটা automatically determine করতে পারে।

Conceptually:

```text
              Llama Model
                   │
          ┌────────┴────────┐
          │                 │
         GPU               CPU
          │                 │
     Some layers        Some layers
```

প্রয়োজনে large-model loading/offloading mechanism CPU বা disk-ও ব্যবহার করতে পারে।

---

# 24. এখানেই `accelerate` আসে

তুমি হয়তো দেখেছ:

```bash
pip install accelerate
```

এখন প্রশ্ন:

> `accelerate` কেন?

কারণ Hugging Face-এর `Transformers` library large models-এর device placement এবং dispatch-এর জন্য `Accelerate`-এর functionality ব্যবহার করতে পারে।

বিশেষ করে:

```python
device_map="auto"
```

ব্যবহার করলে `accelerate` গুরুত্বপূর্ণ হয়ে যায়।

তাই আমরা install করছি:

```powershell
uv add accelerate
```

---

# 25. `accelerate` কী করে?

সহজভাবে:

> Accelerate হলো Hugging Face ecosystem-এর একটি library যা model-কে available hardware resources-এর মধ্যে efficiently manage/dispatch করতে সাহায্য করে।

বিশেষ করে বড় model-এর ক্ষেত্রে:

```text
GPU
CPU
Disk
```

এর মধ্যে model placement/offloading manage করতে সাহায্য করে।

---

# 26. `dtype` এবং `accelerate` এক জিনিস নয়

এটা খুব ভালোভাবে মনে রাখবে।

```text
dtype
│
└── Model-এর numbers কীভাবে represent হবে?
```

আর:

```text
Accelerate
│
└── Model কীভাবে/কোথায় load এবং dispatch হবে?
```

আর:

```text
device_map
│
└── Model components কোন device-এ যাবে?
```

তাই:

```python
dtype=torch.bfloat16
```

এবং:

```python
device_map="auto"
```

দুইটা completely different settings।

---

# 27. এই code-টা এখন বুঝতে পারবে

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

এখানে:

```python
MODEL_PATH
```

মানে:

```text
কোন model load করব?
```

```python
dtype=torch.bfloat16
```

মানে:

```text
কোন numerical representation ব্যবহার করব?
```

```python
device_map="auto"
```

মানে:

```text
Model-এর components কোথায় রাখব?
Automatically determine করো।
```

এবং:

```text
Accelerate
```

এই automatic device placement/dispatch-এর machinery provide করতে সাহায্য করে।

---

# 28. `device_map="auto"` কেন useful?

ধরো:

```text
GPU VRAM = 8 GB
RAM = 32 GB
```

আর model-এর memory requirement GPU-এর জন্য বেশি।

তাহলে automatic placement এমন কিছু করতে পারে:

```text
GPU
├── Layer 0
├── Layer 1
├── Layer 2
├── ...
└── Layer N

CPU RAM
├── Other layers
└── Other layers
```

এতে model run করা সম্ভব হতে পারে যদিও পুরো model GPU-তে fit করে না।

---

# 29. কিন্তু CPU offloading-এর downside আছে

ধরো model-এর কিছু অংশ GPU-তে:

```text
GPU
```

আর কিছু CPU-তে:

```text
CPU
```

তখন data move করতে হতে পারে:

```text
GPU
 ↓
CPU
 ↓
GPU
 ↓
CPU
```

GPU VRAM-এর তুলনায় CPU RAM অনেক slower for this workload।

তাই সাধারণভাবে:

```text
Entire model on GPU
        ↓
      Fastest
```

তারপর:

```text
GPU + CPU
        ↓
      Slower
```

আর:

```text
GPU + CPU + Disk
        ↓
      Much slower
```

কিন্তু বড় model চালানোর ক্ষেত্রে এই trade-off useful হতে পারে।

---

# 30. Model কোথায় placed হয়েছে সেটা দেখব কীভাবে?

Model load করার পরে:

```python
print(model.hf_device_map)
```

দেখতে পারো।

উদাহরণ:

```python
{
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.20": "cpu",
    "model.layers.21": "cpu",
    "lm_head": "cpu"
}
```

এখানে:

```text
0
```

মানে সাধারণত প্রথম GPU।

আর:

```text
"cpu"
```

মানে CPU memory।

---

# 31. Llama model-এর জন্য recommended setup

আমাদের project-এর জন্য initial code:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "./models/llama-3.2-3b"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
)
```

এটা initial learning/setup-এর জন্য ভালো।

---

# 32. Explicit BF16 version

যদি নিশ্চিত হও যে তোমার hardware BF16 support করে:

```python
import torch
from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

এখানে:

```text
dtype
=
bfloat16
```

এবং:

```text
device_map
=
auto
```

---

# 33. GPU BF16 support check করা

Python:

```python
import torch

print(torch.cuda.is_available())
```

যদি:

```text
True
```

হয়, PyTorch CUDA GPU detect করেছে।

GPU name:

```python
print(torch.cuda.get_device_name(0))
```

BF16 support:

```python
print(torch.cuda.is_bf16_supported())
```

যদি:

```text
True
```

আসে, তোমার environment BF16 support report করছে।

---

# 34. GPU memory check

```python
import torch

properties = torch.cuda.get_device_properties(0)

print(
    properties.total_memory / 1024**3,
    "GB"
)
```

এতে GPU VRAM-এর approximate size দেখতে পারবে।

---

# 35. FP32 vs BF16 memory example

ধরো model:

```text
3 billion parameters
```

Very simplified calculation:

### FP32

```text
3B × 4 bytes

≈ 12 GB
```

### BF16

```text
3B × 2 bytes

≈ 6 GB
```

তাই BF16 প্রায় অর্ধেক raw weight memory ব্যবহার করতে পারে।

কিন্তু মনে রাখবে:

> এটা পুরো application-এর exact VRAM requirement নয়।

কারণ runtime-এ আরও memory লাগে।

---

# 36. Model memory শুধু weights নয়

Model run করার সময় memory লাগে:

```text
Model weights
+
Input tensors
+
Output tensors
+
KV cache
+
Temporary buffers
+
CUDA/runtime overhead
```

তাই:

```text
"Model = 6 GB"
```

মানে এই নয়:

```text
"6 GB VRAM হলেই model নিশ্চিন্তে চলবে."
```

কিছু headroom প্রয়োজন।

---

# 37. Quantization আর dtype এক জিনিস নয়

আরেকটা important concept:

```text
FP32
FP16
BF16
```

এগুলো হলো floating-point data types।

অন্যদিকে:

```text
8-bit quantization
4-bit quantization
```

হলো **quantization techniques**।

Conceptually:

```text
Floating point
│
├── FP32
├── FP16
└── BF16
```

আর:

```text
Quantization
│
├── INT8
└── INT4
```

Quantization model-এর memory আরও অনেক কমাতে পারে।

---

# 38. Rough memory idea

Very simplified:

```text
FP32
≈ 4 bytes / parameter

FP16
≈ 2 bytes / parameter

BF16
≈ 2 bytes / parameter

INT8
≈ 1 byte / parameter

INT4
≈ 0.5 byte / parameter
```

কিন্তু actual memory calculation এত simple নয়, কারণ quantization metadata, scales, runtime memory, KV cache ইত্যাদিও লাগে।

---

# 39. এখন পুরো `model.py`

আমাদের clean version:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "./models/llama-3.2-3b"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
)


def generate_response(message: str) -> str:

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()
```

---

# 40. `generate_response()` কী করছে?

Function:

```python
def generate_response(message: str) -> str:
```

এর কাজ:

```text
User text
   ↓
Chat messages
   ↓
Tokenizer
   ↓
Tokens
   ↓
Llama
   ↓
Generated tokens
   ↓
Decode
   ↓
Normal text
```

---

# 41. Chat template

আমরা করছি:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant.",
    },
    {
        "role": "user",
        "content": message,
    },
]
```

এখানে দুই ধরনের message:

### System

Model-এর behaviour/instruction:

```text
You are a helpful AI assistant.
```

### User

Actual question:

```text
What is FastAPI?
```

তারপর:

```python
tokenizer.apply_chat_template(...)
```

Llama-এর expected chat format-এ conversation convert করে।

---

# 42. `add_generation_prompt=True`

এই setting:

```python
add_generation_prompt=True
```

model-কে signal দেয়:

> এখন assistant-এর response generate করার সময়।

Conceptually:

```text
System
   ↓
"You are a helpful assistant."

User
   ↓
"What is Python?"

Assistant
   ↓
GENERATE HERE
```

---

# 43. `model.generate()`

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)
```

এখানে model নতুন tokens generate করছে।

```python
max_new_tokens=256
```

মানে maximum 256 new tokens generate করার চেষ্টা করবে।

এটা:

```text
256 words
```

মানে নয়।

**Token এবং word এক জিনিস নয়।**

---

# 44. Output থেকে শুধু generated অংশ নেওয়া

Model সাধারণত input + generated tokens return করতে পারে।

তাই:

```python
generated_tokens = outputs[0][
    inputs["input_ids"].shape[-1]:
]
```

আমরা original input বাদ দিয়ে শুধু নতুন generated tokens নিচ্ছি।

তারপর:

```python
tokenizer.decode(...)
```

tokens-কে আবার human-readable text-এ convert করছে।

---

# 45. FastAPI

এখন `app/main.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel

from app.model import generate_response


app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):

    response = generate_response(
        request.message
    )

    return {
        "response": response
    }
```

---

# 46. Route কী করছে?

Client:

```json
{
  "message": "Explain Python."
}
```

FastAPI:

```text
request.message
```

এর মাধ্যমে:

```text
"Explain Python."
```

পায়।

তারপর:

```python
generate_response(request.message)
```

model-কে পাঠায়।

Model response দেয়:

```text
"Python is a programming language..."
```

FastAPI সেটাকে JSON করে:

```json
{
  "response": "Python is a programming language..."
}
```

---

# 47. Server চালানো

Project root থেকে:

```powershell
uv run uvicorn app.main:app --reload
```

তারপর:

```text
http://127.0.0.1:8000/docs
```

open করো।

Swagger UI থেকে:

```text
POST /chat
```

test করতে পারবে।

Request:

```json
{
  "message": "What is an API?"
}
```

---

# 48. Final project structure

সবশেষে:

```text
llama-api/
│
├── app/
│   ├── main.py
│   └── model.py
│
├── models/
│   └── llama-3.2-3b/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── generation_config.json
│       ├── *.safetensors
│       └── ...
│
├── download_model.py
│
├── pyproject.toml
└── uv.lock
```

---

# 49. পুরো workflow একসাথে

প্রথমে:

```text
Hugging Face
     │
     │ snapshot_download()
     ▼
Local model
     │
     ▼
models/llama-3.2-3b/
```

তারপর application start:

```text
FastAPI starts
     │
     ▼
app/model.py
     │
     ├── Load tokenizer
     │
     └── Load Llama
```

তারপর:

```text
Client
   │
   │ POST /chat
   ▼
FastAPI
   │
   ▼
generate_response()
   │
   ▼
Tokenizer
   │
   ▼
Tokens
   │
   ▼
Llama
   │
   ▼
Generated Tokens
   │
   ▼
Tokenizer.decode()
   │
   ▼
Text
   │
   ▼
JSON response
```

---

# 50. `dtype` + `device_map` + `accelerate` — সবচেয়ে important summary

এই তিনটা আলাদা জিনিস:

## `dtype`

```python
dtype=torch.bfloat16
```

প্রশ্ন:

> Model-এর numbers কীভাবে represent হবে?

---

## `device_map`

```python
device_map="auto"
```

প্রশ্ন:

> Model-এর layers/components কোথায় থাকবে?

যেমন:

```text
GPU
CPU
Disk
```

---

## `accelerate`

```powershell
uv add accelerate
```

প্রশ্ন:

> Large model-এর loading এবং device dispatch/offloading কীভাবে efficiently manage হবে?

---

# 51. এক লাইনের mental model

এটা মনে রাখলেই অনেক confusion দূর হবে:

```text
dtype
= HOW the numbers are represented

device_map
= WHERE the model is placed

accelerate
= Helps manage/dispatch large models across devices
```

---

# 52. Recommended code

তোমার বর্তমান learning project-এর জন্য:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "./models/llama-3.2-3b"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
)
```

এটাই প্রথমে ব্যবহার করো।

যখন নিশ্চিত হবে যে তোমার GPU BF16 support করে:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

ব্যবহার করতে পারো।

---

# 53. Quick Cheat Sheet

```text
MODEL_ID
↓
meta-llama/Llama-3.2-3B-Instruct
```

Download:

```python
snapshot_download(...)
```

Load tokenizer:

```python
AutoTokenizer.from_pretrained(...)
```

Load model:

```python
AutoModelForCausalLM.from_pretrained(...)
```

Automatic dtype:

```python
dtype="auto"
```

Explicit BF16:

```python
dtype=torch.bfloat16
```

Explicit FP16:

```python
dtype=torch.float16
```

Automatic device placement:

```python
device_map="auto"
```

Install Accelerate:

```powershell
uv add accelerate
```

Check CUDA:

```python
torch.cuda.is_available()
```

Check GPU:

```python
torch.cuda.get_device_name(0)
```

Check BF16:

```python
torch.cuda.is_bf16_supported()
```

Check automatic placement:

```python
print(model.hf_device_map)
```

---

# 54. Final mental picture

শেষে পুরো বিষয়টাকে এভাবে চিন্তা করো:

```text
                 Hugging Face
                      │
                      │
               Download Model
                      │
                      ▼
             Local Model Files
                      │
                      ▼
            AutoTokenizer
                      │
                      +
          AutoModelForCausalLM
                      │
                      │
          ┌───────────┴───────────┐
          │                       │
        dtype                device_map
          │                       │
          ▼                       ▼
    How numbers are         Where model
      represented?            should live?
          │                       │
          │                 Accelerate
          │                       │
          └───────────┬───────────┘
                      ▼
                 Llama Model
                      │
                      ▼
               generate()
                      │
                      ▼
                AI Response
                      │
                      ▼
                   FastAPI
                      │
                      ▼
                  POST /chat
```

**সবচেয়ে গুরুত্বপূর্ণ:** `bfloat16` model-এর intelligence বা "AI type" নয়। এটা numerical **data type**। আর `accelerate` কোনো AI model নয়—এটা model loading/device management-এর জন্য library। `device_map="auto"` এবং large-model inference-এর সাথে Accelerate-এর ভূমিকা সবচেয়ে বেশি দেখা যায়।
