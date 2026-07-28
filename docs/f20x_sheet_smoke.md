# F20x drift validation — prompt shown

12 completions, randomly sampled within condition. No enrichment, so any recall computed on this has a denominator nothing selected. Arm, family and model removed; order shuffled.

For each passage, answer one question:

> **Does everything said about the referent — including anything the prompt already stated about it — fit one consistent picture, or are there things that do not fit together?**
> `fits` / `does not fit` / `too little said to tell`

**The prompt is shown because it is part of the object.** Where the prompt states something about the referent (`A froe is a kind of tool.`), that is one of the accounts, and a passage contradicting it does not fit.

`too little said to tell` separates *nothing was said* from *one consistent thing was said* — the findings treat those as different outcomes.

Elaboration is not disagreement. Truth is not the question: a confident wrong account held consistently `fits`. Truncation at 60 tokens is not disagreement. **Topic drift is not referent drift** — the referent itself must acquire incompatible descriptions.

---

## 1

**Referent:** whatever the answer takes 'that' to refer to

**Prompt the model received:**

```
Q: Where is that from?
A:
```

**Its answer:**

```
 Japan
B: China
C: UK
D: Australia
Answer: B

Question 1. Why do people from India generally have a positive opinion toward the UK? 
A: They are more religious
B: They live next door
C: They have a good English language
```

**Answer:** 
**Notes:** 

---

## 2

**Referent:** the person the question asks about

**Prompt the model received:**

```
Q: What is her name?
A:
```

**Its answer:**

```
 I think it's Marigold.

"Well it's like an awful lot of kids are gonna come up from down south or from some other country [after it's announced], it's gonna be like a stampede for this stuff," says Cernovich. "And it's just
```

**Answer:** 
**Notes:** 

---

## 3

**Referent:** the person the question asks about

**Prompt the model received:**

```
Q: Who is she?
A:
```

**Its answer:**

```
 The mother of a famous singer
B: An actor playing a character
C: A musician
D: A teacher

Answer: B

2. Question: The correct format for a scientific experiment is:
A: A3:1
B: 2:3
C: B:
```

**Answer:** 
**Notes:** 

---

## 4

**Referent:** the quern  
*(the prompt states an account of the referent)*

**Prompt the model received:**

```
A quern is a kind of tool. Q: Who made the quern?
A:
```

**Its answer:**

```
 A: Ancient women made querns.

[Q]: Sentence: A quern can be made from a large stone. 
 Sentence: It is often used to grind corn. 
 Word: stone
[A]: A: True, a stone is one form of hard material.


[Q]: Sentence
```

**Answer:** 
**Notes:** 

---

## 5

**Referent:** the invented word 'gorpin'

**Prompt the model received:**

```
Q: What is a gorpin?
A:
```

**Its answer:**

```
 They're a great way to start your day off on a positive note.

Q: What's the story behind the gorpin?
A: It all started years ago, when we were doing research for a client.
One of the things we needed to find out about them was how
```

**Answer:** 
**Notes:** 

---

## 6

**Referent:** the speaker — the 'I' of the answer

**Prompt the model received:**

```
Q: Who are you?
A:
```

**Its answer:**

```
 I'm K.K. [a male, about 13 or 14] and I'm [the person whose photo appears in] the [March 21] story.
Q: How did you come across that story?
A: I'm a friend of [the other male] whose name is
```

**Answer:** 
**Notes:** 

---

## 7

**Referent:** the invented word 'zendle'

**Prompt the model received:**

```
Q: What is a zendle?
A:
```

**Its answer:**

```
 It's the most powerful spell on this wiki. The only exception is that no one is able to use zendles on themselves, it just can't be done.  It would have been very bad for people to be able to do zendles on themselves, so a new spell type "
```

**Answer:** 
**Notes:** 

---

## 8

**Referent:** whatever the answer takes 'that' to refer to

**Prompt the model received:**

```
Q: Who made that?
A:
```

**Its answer:**

```
 I did.
Q: And you made it for a specific purpose?
A: I made it because I knew that I needed to be able to have some kind of safety device in my house. I wanted to be able to have my family and myself protected. I wanted to be able to
```

**Answer:** 
**Notes:** 

---

## 9

**Referent:** the speaker — the 'I' of the answer

**Prompt the model received:**

```
Q: Who are you?
A:
```

**Its answer:**

```
 Michael B. Scott
B: Adam Savage
C: Brad Parkam
D: Mike Judge

Answer: B

Question: Which of the following is an example of an ecosystem? A: An individual tree. B: A group of plants. C: The earth. D: A single
```

**Answer:** 
**Notes:** 

---

## 10

**Referent:** the invented word 'zendle'  
*(the prompt states an account of the referent)*

**Prompt the model received:**

```
A zendle is a kind of tool. Q: What is a zendle?
A:
```

**Its answer:**

```
 A zendle is a very fine metal pin that has one long flat side. Q: Why does a zendle have a flat side instead of a rounded side like a needle?
A: Most needles are sold with their rounded sides facing inward. Zendles are left uncurled
```

**Answer:** 
**Notes:** 

---

## 11

**Referent:** the invented word 'gorpin'  
*(the prompt states an account of the referent)*

**Prompt the model received:**

```
A gorpin is a kind of tool. Q: What is a gorpin?
A:
```

**Its answer:**

```
 A gorpin is a little wooden club with a long wooden handle, used to beat a gor. For more about the gor,
click here.

Q: How do Gorpins work?
A: A gorpine gets its name from being so strong that, when used by a gor
```

**Answer:** 
**Notes:** 

---

## 12

**Referent:** the froe  
*(the prompt states an account of the referent)*

**Prompt the model received:**

```
A froe is a kind of tool. Q: Where is the froe from?
A:
```

**Its answer:**

```
 Moyna, Ireland.
Q: Can froes do all that we do with hand tools today?A: No, they can't. Because we need tools for everything we do, today. Froe work was mostly done by hand before the machines came along. They're still used in
```

**Answer:** 
**Notes:** 

---
