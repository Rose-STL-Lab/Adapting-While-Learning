import random
import re
import os
import transformers
import torch
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer

instruction = r"""
You are a teacher specialized in scientist. Your task is to provide a detailed solution process for a given questions. The question was answered with relevant code and logical reasoning, but you shouldn't mention these tools using procedures in your answer. The final answer shoud be presented in latex like '\boxed{\frac{1}{2}}'.

You should provide a solution process that includes:
- Relevant data from the climate simulator (but you shouldn't mention the coding, you should directly list relevant data)
- Detailed analysis of this data
- Logical reasoning connecting the data to the question
- Explain with equation and formula, write like a scientist.

If you think the answer is wrong, you should answer 'The answer is wrong' and explain why.

Your answer should be in the following format:
Solution: <solution>
Answer: <answer, a pure number>

Example1:

User:
Question: What is the arithmetic mean of all of the positive two-digit integers with the property that the integer is equal to the sum of its first digit plus its second digit plus the product of its two digits?

Following is the interaction for answering the question.
{
    "role": "assistant",
    "content": "Thought: To find the arithmetic mean of all the positive two-digit integers with the given property, we need to find all the two-digit numbers that satisfy the condition and then calculate their average.\nAction: write_and_run_code\nCode:\n```python\ndef find_two_digit_numbers():\n    valid_numbers = []\n    for i in range(10, 100):\n        # Extract the first and second digits\n        first_digit = i // 10\n        second_digit = i % 10\n\n        # Check if the number is equal to the sum of its digits plus the product of its digits\n        if i == first_digit + second_digit + first_digit * second_digit:\n            valid_numbers.append(i)\n\n    return valid_numbers\n\nvalid_numbers = find_two_digit_numbers()\nprint(\"Valid numbers:\", valid_numbers)\n\ndef calculate_arithmetic_mean(numbers):\n    return sum(numbers) / len(numbers)\n\narithmetic_mean = calculate_arithmetic_mean(valid_numbers)\nprint(\"Arithmetic mean:\", arithmetic_mean)\n```"
},
{
    "role": "user",
    "content": "Feedback: Output:\nValid numbers: [19, 29, 39, 49, 59, 69, 79, 89, 99]\nArithmetic mean: 59.0\n"
},
{
    "role": "assistant",
    "content": "Thought: The code successfully identified the valid two-digit numbers that satisfy the given property and calculated their arithmetic mean. The valid numbers are indeed [19, 29, 39, 49, 59, 69, 79, 89, 99], and their arithmetic mean is 59.0.\nAction: answer_question\nSolution: The arithmetic mean of all the positive two-digit integers with the property that the integer is equal to the sum of its first digit plus its second digit plus the product of its two digits is 59.\n\nAnswer: 59"
}

Assistant:
Solution: To find the arithmetic mean of all positive two-digit integers that satisfy the property where the integer is equal to the sum of its first digit, second digit, and the product of its two digits, we start by identifying such numbers.

Let the two-digit number be represented as:

\[
N = 10a + b
\]

where \(a\) is the first digit (tens place) and \(b\) is the second digit (ones place). The given condition is:

\[
N = a + b + ab
\]

Substituting \(N = 10a + b\) into the equation:

\[
10a + b = a + b + ab
\]

Simplify the equation:

\[
10a + b - b = a + ab
\]
\[
10a = a + ab
\]
\[
10a - a = ab
\]
\[
9a = ab
\]
\[
b = \frac{9a}{a} = 9
\]

Thus, for each \(a\) (the tens digit), the second digit \(b\) must be 9. This gives us the numbers:

\[
19, 29, 39, 49, 59, 69, 79, 89, 99
\]

Now, let's calculate the arithmetic mean. The arithmetic mean of a set of numbers is given by:

\[
\text{Mean} = \frac{\text{Sum of the numbers}}{\text{Number of numbers}}
\]

The sum of the valid numbers is:

\[
19 + 29 + 39 + 49 + 59 + 69 + 79 + 89 + 99 = 531
\]

The number of valid numbers is 9. Therefore, the arithmetic mean is:

\[
\text{Mean} = \frac{531}{9} = 59
\]

Answer: \boxed{59}

Example 2:
User:
Problem: In how many ways can seven beads of distinct colors be put on the hexagonal grid shown, if reflections and rotations of an arrangement are considered equivalent?\n\n[asy]\nsize(50);\ndot((0,0));\ndot((1,0));\ndot((-1,0));\ndot((.5,sqrt(3)/2));\ndot((.5,-sqrt(3)/2));\ndot((-.5,sqrt(3)/2));\ndot((-.5,-sqrt(3)/2));\n[/asy]

Following is the interaction for answering the question.
{
    "role": "assistant",
    "content": "Thought: The problem requires counting the distinct arrangements of seven beads of different colors on a hexagonal grid. We will consider rotations and reflections as equivalent arrangements.\n\nAction: write_and_run_code\nCode:\n```python\nimport math\n\n# Number of distinct arrangements without considering rotations and reflections\ndistinct_arrangements_without_symmetry = math.factorial(7)\n\n# The hexagonal grid has a symmetry group of order 12\nsymmetry_group_order = 12\n\n# Number of distinct arrangements considering rotations and reflections\ndistinct_arrangements_with_symmetry = distinct_arrangements_without_symmetry / symmetry_group_order\n\nprint(\"Number of distinct arrangements considering rotations and reflections:\", distinct_arrangements_with_symmetry)\n```"
},
{
    "role": "user",
    "content": "Feedback: Output:\nNumber of distinct arrangements considering rotations and reflections: 420.0\n"
},
{
    "role": "assistant",
    "content": "Thought: The code correctly calculates the number of distinct arrangements of the beads on the hexagonal grid, considering rotations and reflections as equivalent. The total number of distinct arrangements without considering symmetry is 7!, and the hexagonal grid has a symmetry group of order 12. Therefore, the number of distinct arrangements considering rotations and reflections is 7! / 12.\n\nAction: answer_question\nSolution: To find the number of distinct arrangements of the beads on the hexagonal grid, we need to consider the symmetry of the grid. The grid has a hexagonal symmetry, which means it looks the same under rotations of 60, 120, 180, 240, and 300 degrees. There are 6 such rotations, and also 6 reflections (3 along the x-axis and 3 along the y-axis). Therefore, the total number of symmetries is 6 rotations + 6 reflections = 12.\n\nSince there are 7 distinct beads, the total number of arrangements without considering symmetry is 7! = 5040. However, we need to divide this by the number of symmetries (12) to get the number of distinct arrangements considering rotations and reflections.\n\nTherefore, the number of distinct arrangements is 5040 / 12 = 420.\n\nAnswer: 420"
}

Assistant:
Solution: To determine how many distinct ways we can arrange seven beads of different colors on a hexagonal grid, accounting for rotational and reflectional symmetry, we proceed as follows.

### Step 1: Total number of arrangements without symmetry
If we disregard symmetry, the number of ways to arrange 7 distinct beads on 7 positions is simply the number of permutations of 7 objects, which is:

\[
7! = 7 \times 6 \times 5 \times 4 \times 3 \times 2 \times 1 = 5040
\]

### Step 2: Accounting for symmetry
The hexagonal grid possesses symmetry. Specifically, it has 6 rotational symmetries (rotations by \(60^\circ\), \(120^\circ\), \(180^\circ\), \(240^\circ\), and \(300^\circ\) plus the identity rotation) and 6 reflectional symmetries (3 reflections across axes and 3 across lines of symmetry between axes). Therefore, the symmetry group of the hexagon has 12 elements in total.

### Step 3: Applying Burnside's Lemma
To account for these symmetries, we divide the total number of arrangements by the size of the symmetry group (12). This gives the number of distinct arrangements, accounting for rotations and reflections:

\[
\frac{7!}{12} = \frac{5040}{12} = 420
\]

Answer: \boxed{420}

Example3:
User:
Problem: Let $X$ equal the maximal oxygen intake of a human on a treadmill, where the measurements are in milliliters of oxygen per minute per kilogram of weight. Assume that, for a particular population, the mean of $X$ is $\\mu=$ 54.030 and the standard deviation is $\\sigma=5.8$. Let $\\bar\{X\}$ be the sample mean of a random sample of size $n=47$. Find $P(52.761 \leq \bar\{X\} \leq 54.453)$, approximately. The unit is .

{'role': 'user', 'content': 'Problem: Let $X$ equal the maximal oxygen intake of a human on a treadmill, where the measurements are in milliliters of oxygen per minute per kilogram of weight. Assume that, for a particular population, the mean of $X$ is $\\mu=$ 54.030 and the standard deviation is $\\sigma=5.8$. Let $\\bar{X}$ be the sample mean of a random sample of size $n=47$. Find $P(52.761 \\leq \\bar{X} \\leq 54.453)$, approximately.  The unit is  .'}
{'role': 'assistant', 'content': 'Thought: We are given the population mean ($\\mu$), population standard deviation ($\\sigma$), and the sample size ($n$). We need to find the probability that the sample mean ($\\bar{X}$) lies within a certain range. Since we are dealing with sample means, we can use the Central Limit Theorem (CLT), which states that the distribution of sample means will be approximately normal with a mean equal to the population mean ($\\mu$) and a standard deviation equal to the population standard deviation divided by the square root of the sample size ($\\sigma/\\sqrt{n}$). We can standardize the given values to use the standard normal distribution (Z-table) to find the probability.\n\nAction: write_and_run_code\nCode:\n```python\nimport math\nfrom scipy.stats import norm\n\n Given values\nmu = 54.030  # Population mean\nsigma = 5.8  # Population standard deviation\nn = 47  # Sample size\n\n# Calculate the standard deviation of the sampling distribution\nstd_dev_sampling_dist = sigma / math.sqrt(n)\n\n# Given range\nlower_bound = 52.761\nupper_bound = 54.453\n\n# Standardize the values\nz_lower = (lower_bound - mu) / std_dev_sampling_dist\nz_upper = (upper_bound - mu) / std_dev_sampling_dist\n\n# Use the Z-table to find the probabilities\nprob_lower = norm.cdf(z_lower)\nprob_upper = norm.cdf(z_upper)\n\n# The probability that the sample mean lies within the range is the difference between the upper and lower probabilities\nprob = prob_upper - prob_lower\n\nprint(f"P({lower_bound} <= \\bar{X} <= {upper_bound}) = {prob}")\n```'}
{'role': 'user', 'content': 'Feedback: \n\nError:\nTraceback (most recent call last):\n  File "", line 27, in <module>\n    print(f"P({lower_bound} <= \\bar{X} <= {upper_bound}) = {prob}")\nNameError: name \'X\' is not defined\n'}
{'role': 'assistant', 'content': 'Thought: The error message indicates that there is an undefined variable \'X\'. However, I don\'t see any variable named \'X\' in the code. Upon reviewing the code, I realize that I missed the variable \'X\' in the print statement. The correct code should print the probability as P(52.761 <= X <= 54.453) instead of P(52.761 <= X <= 54.453) = prob.\n\nAction: write_and_run_code\nCode:\n```python\nimport math\nfrom scipy.stats import norm\n\n# Given values\nmu = 54.030  # Population mean\nsigma = 5.8  # Population standard deviation\nn = 47  # Sample size\n\n# Calculate the standard deviation of the sampling distribution\nstd_dev_sampling_dist = sigma / math.sqrt(n)\n\n# Given range\nlower_bound = 52.761\nupper_bound = 54.453\n\n# Standardize the values\nz_lower = (lower_bound - mu) / std_dev_sampling_dist\nz_upper = (upper_bound - mu) / std_dev_sampling_dist\n\n# Use the Z-table to find the probabilities\nprob_lower = norm.cdf(z_lower)\nprob_upper = norm.cdf(z_upper)\n\n# The probability that the sample mean lies within the range is the difference between the upper and lower probabilities\nprob = prob_upper - prob_lower\n\nprint(f"P({lower_bound} <= X <= {upper_bound}) = {prob}")\n```'}
{'role': 'user', 'content': 'Feedback: Output:\nP(52.761 <= X <= 54.453) = 0.6246478404435203\n'}
{'role': 'assistant', 'content': 'Thought: The code is now working correctly and printing the desired probability. However, the problem statement asks for an approximate answer. Since the output is already an approximation (due to the use of the Z-table), we can provide the final answer as a decimal number without any special characters.\n\nAction: answer_question\nSolution: The desired probability is approximately 0.6246.\nAnswer: 0.6246'}

Assistant:
Solution:
We aim to determine the probability that the sample mean \( \bar{X} \) of maximal oxygen intake lies within the interval \( 52.761 \leq \bar{X} \leq 54.453 \) milliliters of oxygen per minute per kilogram of weight, given the population parameters.

**Given Data:**
- **Population Mean (\( \mu \)):** 54.030 ml/min/kg
- **Population Standard Deviation (\( \sigma \)):** 5.8 ml/min/kg
- **Sample Size (\( n \)):** 47

**Analysis:**

1. **Sampling Distribution of the Sample Mean (\( \bar{X} \)):**
   - According to the Central Limit Theorem, for a sufficiently large sample size (\( n \geq 30 \)), the sampling distribution of the sample mean \( \bar{X} \) is approximately normally distributed.
   - **Mean of \( \bar{X} \) (\( \mu_{\bar{X}} \)):**
     \[
     \mu_{\bar{X}} = \mu = 54.030 \, \text{ml/min/kg}
     \]
   - **Standard Deviation of \( \bar{X} \) (\( \sigma_{\bar{X}} \)):**
     \[
     \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} = \frac{5.8}{\sqrt{47}} \approx \frac{5.8}{6.8557} \approx 0.847 \, \text{ml/min/kg}
     \]

2. **Standardizing the Desired Interval:**
   - To find \( P(52.761 \leq \bar{X} \leq 54.453) \), we convert the sample mean values to their corresponding z-scores using the formula:
     \[
     Z = \frac{\bar{X} - \mu_{\bar{X}}}{\sigma_{\bar{X}}}
     \]
   - **Lower Bound (\( \bar{X} = 52.761 \)):**
     \[
     Z_{\text{lower}} = \frac{52.761 - 54.030}{0.847} \approx \frac{-1.269}{0.847} \approx -1.50
     \]
   - **Upper Bound (\( \bar{X} = 54.453 \)):**
     \[
     Z_{\text{upper}} = \frac{54.453 - 54.030}{0.847} \approx \frac{0.423}{0.847} \approx 0.50
     \]

3. **Determining the Probability:**
   - Using standard normal distribution tables or software, we find the probabilities corresponding to the z-scores.
   - **Probability Corresponding to \( Z_{\text{lower}} = -1.50 \):**
     \[
     P(Z \leq -1.50) \approx 0.0668
     \]
   - **Probability Corresponding to \( Z_{\text{upper}} = 0.50 \):**
     \[
     P(Z \leq 0.50) \approx 0.6915
     \]
   - **Probability that \( \bar{X} \) Lies Between 52.761 and 54.453:**
     \[
     P(52.761 \leq \bar{X} \leq 54.453) = P(Z \leq 0.50) - P(Z \leq -1.50) \approx 0.6915 - 0.0668 = 0.6247
     \]

**Conclusion:**
The probability that the sample mean maximal oxygen intake \( \bar{X} \) lies between 52.761 and 54.453 ml/min/kg is approximately **0.6247**.

Answer: \boxed{0.6247}

Now Begin!
"""

MODEL_PF = ""


def process_chunk(chunk, gpu_id, model_id, instruction):
    gpu_id = gpu_id % torch.cuda.device_count()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Initialize vLLM and tokenizer
    llm = LLM(
        model=model_id,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=0.7,
    )

    processed_chunk = []
    prompts = []
    valid_items = []

    for item in chunk:
        if MODEL_PF not in item or not item[MODEL_PF]:
            continue
        if (
            "answer:" not in item[MODEL_PF][-1]["content"].lower()
            or "solution:" not in item[MODEL_PF][-1]["content"].lower()
        ):
            continue

        prompt = (
            "\n\nFollowing is the interaction for answering the question.\n\n"
            + "\n".join([str(i) for i in item[MODEL_PF][-3:]])
        )

        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": "Question: "
                + item["problem"]
                + prompt
                + "Now begin to write solution for this question based on the question and code and answer.",
            },
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompts.append(prompt)
        valid_items.append(item)

    # Batch inference
    outputs = llm.generate(prompts, sampling_params)
    
    for item, output in zip(valid_items, outputs):
        item["cot"] = output.outputs[0].text
        processed_chunk.append(item)

    return processed_chunk

def main():
    model_id = ""
    raw_data = []
    with open("") as f:
        raw_data = json.load(f)
    num_chunks = 7
    chunk_size = len(raw_data) // num_chunks + 1
    chunks = [
        raw_data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks - 1)
    ]
    chunks.append(raw_data[(num_chunks - 1) * chunk_size :])

    print(f"Number of chunks: {len(chunks)}")

    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        futures = [
            executor.submit(process_chunk, chunk, i, model_id, instruction)
            for i, chunk in enumerate(chunks)
        ]

        processed_data = []
        for future in as_completed(futures):
            processed_data.extend(future.result())

    with open("", "w") as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()