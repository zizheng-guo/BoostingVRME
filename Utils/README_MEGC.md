
# MEGC

Step1：**Organize Dataset in the Required Format**

CASME:

```
dataset
├── label.xlsx
└── data
    └── subject
        ├── a1
        └── b1
```

SAMM:

```
dataset
├── label.xlsx
└── dataset
    └── data
        ├── subject_1
        └── subject_2
```

**Note:** Since the code framework requires the input of labels, the mock label file should be created. You can refer to `mock_label/label.xlsx` for the format. 

Step2：**Run the Pre-trained Model**

```bash
python main.py --dataset_name SAMM_test --train False
```

Step3：**Traditional Methods as the Spotting Branch**

Run Traditional Method for Spotting and Filter Out Macro-expressions, and Use Pre-trained Model Inference for Recognition

Step4：**Apply Non-Maximum Suppression (NMS)**

```markdown
Input: Interval pred1 = [a1, b1], pred2 = [a2, b2]
Output: Adjusted interval pred
Procedure:
if b1 - a2 > k_p / 2 then
a3 = a1 + k_p / 4
b3 = b2 - k_p / 4
return pred = [a3, b3]
end if

```

note：Step3 and Step4 Only applicable for CASME, can bring a slight improvement

