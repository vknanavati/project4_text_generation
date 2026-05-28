# Loss and Perplexity

## Loss

Loss measures **how wrong the model's predictions were** on average across every single character prediction in the dataset.

It's calculated using something called **cross entropy** — which essentially asks "how much probability did the model assign to the correct answer?"

- If the model said `'e'` has 90% probability and `'e'` was correct → low loss
- If the model said `'e'` has 2% probability and `'e'` was correct → high loss

In our results:

```
LSTM        final loss: 1.5446
RNN         final loss: 1.6926
Transformer final loss: 1.6885
```

Lower is better. The LSTM had the lowest loss — it was wrong by the least amount on average.

---

## Perplexity

Perplexity is just loss converted into a more intuitive scale. The formula is:

```
perplexity = e^loss
```

It answers the question: **"On average, how many characters was the model effectively choosing between at each step?"**

Using our actual results:

```
LSTM:        perplexity 4.69  → choosing between ~5 characters at each step
RNN:         perplexity 5.43  → choosing between ~5-6 characters at each step
Transformer: perplexity 5.41  → choosing between ~5-6 characters at each step
```

Compare that to a completely untrained model:

```
Untrained: perplexity ~65 → choosing randomly between all 65 characters
```

So our LSTM went from being completely clueless (65 options) to being fairly confident (about 5 options) after just 3 epochs. That's meaningful learning.

---

## Why do we have both?

They measure the same thing — just expressed differently:

| Metric | What it tells you | Scale |
|---|---|---|
| Loss | Raw mathematical error | Lower is better, no upper bound |
| Perplexity | How many choices the model felt it had | Lower is better, starts at vocab size |

**Analogy:** Loss is like a raw exam score out of 1000 points. Perplexity is like converting that to a letter grade — same underlying information, just expressed in a way that's easier to interpret and compare across different models and datasets.