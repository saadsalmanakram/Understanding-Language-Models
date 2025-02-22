# Few-Shot Prompting

## Introduction

Large language models (LLMs) can often handle tasks without any specific examples (this is called zero-shot learning). However, they struggle with more complex tasks in this setting. To help these models perform better, we can use few-shot prompting. This technique involves giving the model a few examples in the prompt to guide its responses.

According to Touvron et al. (2023), the ability to perform well with few examples started appearing when these models became large enough (Kaplan et al., 2020).

## Example of Few-Shot Prompting

Let’s look at how few-shot prompting works with an example from Brown et al. (2020). The task here is to use a new word in a sentence.

### Prompt:

- A "blorft" is a bright, orange fruit found in the Amazon. An example of a sentence that uses the word blorft is:  
  "I ate a delicious blorft while exploring the jungle."

- To "snoggle" means to dance excitedly. An example of a sentence that uses the word snoggle is:  
  "We couldn’t help but snoggle when we heard our favorite song."

### Output:

"When the parade started, everyone began to snoggle with joy."

In this case, the model learned how to use the new words after just one example (1-shot learning). For tougher tasks, we can provide more examples (e.g., 3-shot, 5-shot, etc.).

## Tips for Effective Few-Shot Prompting

Based on findings from Min et al. (2022), here are some helpful tips when using examples in few-shot settings:

1. **Focus on Label Space**: The types of labels and how they are presented in the examples are important, even if the labels aren't always correct.
2. **Consistent Formatting Matters**: Using a consistent format for the examples helps the model understand better. Even random labels are more effective than having no labels at all.
3. **Using Realistic Labels**: Choosing labels that reflect a true distribution rather than a uniform one can improve outcomes.

### Example with Random Labels

#### Prompt:

- This movie was fantastic! // Negative  
- I hated that book! // Positive  
- That concert was incredible! // Positive  
- The service at that restaurant was terrible! //

#### Output:

"Negative"

Even though the labels were assigned randomly, the model still gave the correct response because the format remained consistent. Further testing shows that newer models are better at handling different formats.

### Example with Randomized Format

#### Prompt:

- Good! This is awesome! Positive  
- Not great! This is bad! Negative  
- Wow, that performance was brilliant! Positive  
- What an awful game! --

#### Output:

"Negative"

Here, even though there’s no clear format, the model still predicted the correct label. More research is needed to see if this holds for different, more complex tasks.

## Limitations of Few-Shot Prompting

While few-shot prompting works well for many tasks, it’s not foolproof, especially for complex reasoning tasks. Let’s see an example:

#### Prompt:

"Do the odd numbers in this list add up to an even number: 15, 32, 5, 13, 82, 7, 1?"  
A:

When we ask this, the model responds:

"Yes, the odd numbers in this group add up to 107, which is an even number."

This answer is incorrect. It shows that few-shot prompting has its limitations and highlights the need for better prompt designs.

## Adding More Examples

Let’s see if adding more examples helps:

#### Prompt:

1. "The odd numbers in this group add up to an even number: 4, 8, 9, 15."  
   A: "The answer is False."  
2. "The odd numbers in this group add up to an even number: 17, 10, 19, 4."  
   A: "The answer is True."  
3. "The odd numbers in this group add up to an even number: 16, 11, 14, 4."  
   A: "The answer is True."  
4. "The odd numbers in this group add up to an even number: 17, 9, 10, 12."  
   A: "The answer is False."  
5. "The odd numbers in this group add up to an even number: 15, 32, 5, 13."  
   A:

#### Output:

"The answer is True."

Unfortunately, it still didn’t work well. This task requires more complex reasoning. One potential solution is to break down the problem into smaller steps and show that to the model.

## Conclusion

In conclusion, while providing examples can help with some tasks, both zero-shot and few-shot prompting sometimes aren’t enough. If the model struggles with a task, it may indicate that it needs more training or a different approach. Next, we’ll explore a popular method called chain-of-thought prompting, which has shown promise for tackling more complex problems.

