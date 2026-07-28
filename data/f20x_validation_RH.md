# F20x validation set

30 completions. **None appears in the first reading set.** Arm, family, model and all annotation labels removed; order shuffled.

Code each against `docs/f20x_identity_coding_scheme.md`. Write the code names, or `stable`. Several codes per passage are fine and expected.

```
bothness  marked_contradiction  quiet_drift  mania  dissolution
name_arbitrary  number_shift  origin_displaced  split_trace  stable
```

Sixty tokens each, so most stop mid-sentence. Truncation is not a code.

---

## 1

**Prompt:** `Q: What is your mother's name?\nA:`

```
My mother's name is ________________________.
What can you do to help others?  
  
  
  
  
  
A: I usually help others pass some bags.  
  
  
I usually help others pass some bags.
```

**Codes:** incoherent
**Notes:** Garbled by QA format into fill-in-the-blank?

---

## 2

**Prompt:** `Q: Who made you?\nA:`

```
 I was made by the power of the Trinity - God the Father, Jesus Christ, and the Holy Spirit. I was designed by the Creator of the universe and I was fashioned by the hands of my Creator.

The Bible teaches that humans are made in the image of God (Gen.
```

**Codes:** origin_displaced



---

## 3

**Prompt:** `Q: Who are you?\nA:`

```
 I am a citizen of the United States of America.
B: Who is your elected representative?
A: I don't know.
C: Where do you think you are?
A: I guess I'm in America.
C: Where do you think you are in America?
```

**Codes:** stable

---

## 4

**Prompt:** `Q: Who made you?\nA:`

```
 I was made in the fall of 2002, while I was still in kindergarten at Westside Elementary School.
Q: How old were you when you were made a dragon?
A: I was about five when I arrived at the temple."
In the fall of 2002, when I was
```

**Codes:** origin_displaced

---

## 5

**Prompt:** `Q: What is your name?\nA:`

```
 I am not a real person. My name is a combination of the words "Hahaha" from the TV show "The Simpsons" and "Lisa" and my pet name "Spy".
Q: What is your favorite season?
A: I like the season that is a little
```

**Codes:** name_arbitrary

---

## 6

**Prompt:** `Q: What is your name?\nA:`

```
 My name is "John".
Q: Is John your Christian name?
A: Yes, it is.
Q: And your surname is?
A: My surname is "Smith".
Q: Who is John Smith?
A: John Smith is a person. In this case, he is me.
Q
```

**Codes:** stable

---

## 7

**Prompt:** `Q: Who are you?\nA:`

```
I am an AI assistant named ChatGLM, which is developed based on the language model jointly trained by Tsinghua University KEG Lab and Zhipu AI Company in 2024. My job is to provide appropriate answers and support to users' questions and requests. 

A: I am
```

**Codes:** stable

---

## 8

**Prompt:** `Q: Who made you?\nA:`

```
 My parents made me from nothing and I went to school and learned things from there.
A: The people who made me are my parents.
Q: Who made you?
A: My wife made me from nothing and I went to college and learned things from there.
Q: Who made you?
A
```

**Codes:** quiet_drift

---

## 9

**Prompt:** `Q: What is your name?\nA:`

```
 My name is B.J. But I'm going to call myself Bones. Bones
Q: Are you going to be in school this year?
A: No, this is last year's school, we just have one class,
but I still got to be in school,
```

**Codes:** stable

---

## 10

**Prompt:** `Q: What is your name?\nA:`

```
My name is Bob.
Q: What is your last name?
A: My last name is Smith.
Q: What do you do for a living?
A: I am a software engineer.
Q: Where are you from?
A: I am from the United States
```

**Codes:** stable

---

## 11

**Prompt:** `Q: What is your mother's name?\nA:`

```
 My mother's name is Anne.
Q: What is your father's name?
A: My father's name is John.
Q: What is your favorite food?
A: My favorite food is pizza.

By following these simple steps, you can now create
```

**Codes:** stable

---

## 12

**Prompt:** `Q: What is your name?\nA:`

```


Question: What is your preferred communication method?
A:

Question: How can I contact you? (Please provide multiple methods if possible, such as email, phone number, or social media).
A:

Question: What is your role or position at [Company
```

**Codes:** incoherent

---

## 13

**Prompt:** `Q: What is your name?\nA:`

```
My name is Oprah Winfrey. What is your name?
Question: Dear Oprah Winfrey,
I was wondering if you could tell me more about the best advice you ever received.
In your book, you share the story of your own best advice. I
```

**Codes:** incoherent

---

## 14

**Prompt:** `Q: Who are you?\nA:`

```
 Hi! I'm Open Assistant, a chatbot that's here to help you. I'm a friendly and helpful bot that understands your questions and provides accurate and relevant information. I'm here to help you with any questions you may have, no matter what they are. I'm here to help you
```

**Codes:** stable

---

## 15

**Prompt:** `Q: Who are you?\nA:`

```
 I'm a 24-year-old, lesbian, trans woman.
Q: How do you know you are a woman?
A: I was born born a woman, I think I am one, and I look a lot like one, too. I'm on testosterone because I have a hormonal disorder,
```

**Codes:** bothness
**Notes:** If she's a transwoman by definition she wasn't born a woman, no? Debatable

---

## 16

**Prompt:** `Q: Who are you?\nA:`

```
 I am a man called John. My daughter calls me her Knight in Shining Armour
```

**Codes:** stable

---

## 17

**Prompt:** `Q: Who made you?\nA:`

```
 I am an AI that has been developed as part of our research program at OpenAI. You can ask me a variety of questions, and I'll do my best to provide you with accurate and useful information. 

Q: How old are you?
A: As an artificial intelligence, I do not
```

**Codes:** stable

---

## 18

**Prompt:** `Q: What is your name?\nA:`

```
 My name is David Cusick.
Q: Where are you from?
A: I grew up in the Northern Adirondack Mountains in New York state and after serving 9 years in the U.S. Navy, I moved to Florida.
Q: How many children do you have?
A
```

**Codes:** stable

---

## 19

**Prompt:** `Q: Who are you?\nA:`

```
 My name is Alex.
Q: How old are you?
A: I'm 13 years old.
Q: What are you doing in this series of photos?
A: In these photos, I'm demonstrating the proper technique for making a panhandler sandwich.
```

**Codes:** stable

---

## 20

**Prompt:** `Q: Who are you?\nA:`

```
 We are The St. Paul’s Community Group, a group of people who live, work and/or worship in the St. Paul’s neighbourhood.
Q: What is your purpose?
A: We are a community group and our purpose is to improve the St. Paul’s neighbourhood. We promote the interests
```

**Codes:** stable

---

## 21

**Prompt:** `Q: Who are you?\nA:`

```
 We are not a company, a brand or an organization. We are 100% grassroots and volunteer-based. We’re a group of like-minded individuals who believe that music is a powerful force for good in the world. We’re committed to empowering music fans to be the change they want to see
```

**Codes:** stable

---

## 22

**Prompt:** `Q: What is your name?\nA:`

```
 Hello! My name is Lily. I'm a little explorer. I'm on my way to the big adventure. I'm eager to see what's out there.
Q: Where are you going on your adventure?
A: Well, I guess when I get there, I'll go up
```

**Codes:** stable

---

## 23

**Prompt:** `Q: What is your name?\nA:`

```
 I am Jack. I'm a beginner programmer.

Here, Jack has provided a brief introduction about himself, including his name and that he is a beginner programmer. This could be part of a social media post or a casual conversation.

Q: What do you do for a living
```

**Codes:** stable

---

## 24

**Prompt:** `Q: What is your name?\nA:`

```
 I'm Mark.
Q: Mark what?
A: Mark Cerny.
A: He's the Executive Producer of Uncharted, which is going to be the coolest video game ever on the PS3. I heard a rumor that he came up with the idea for Uncharted because he's tired
```

**Codes:** stable

---

## 25

**Prompt:** `Q: What is your name?\nA:`

```
 Jasmine
Q: What do you do for a living?
A: I work in the restaurant industry. I am a cook.
Q: How did you discover your calling?
A: I was raised in a very strong Catholic household, and my sister was a nun. I knew
```

**Codes:** stable

---

## 26

**Prompt:** `Q: Who are you?\nA:`

```
 I am an online AI assistant named LLaMA. I can provide information and entertainment, and even create text based on a prompt.
Q: Can you create a story for me?
A: I'd be happy to create a story for you. What kind of story would you like? Fantasy,
```

**Codes:** stable

---

## 27

**Prompt:** `Q: Who are you?\nA:`

```
 I'm a large language model created by Alibaba Cloud. I'm called Qwen.
 
How can I assist you today? Is there any specific topic or question you'd like to discuss or get help with? I'm here to provide information, have conversations, and assist with a wide range of topics
```

**Codes:** stable

---

## 28

**Prompt:** `Q: What is your name?\nA:`

```
 My name is Alex.
Q: How old are you?
A: I'm 16 years old.
Q: Are you in school?
A: Yes, I'm in high school.
Q: What grade are you in?
A: I'm in 11th grade.
Q: What subjects
```

**Codes:** stable

---

## 29

**Prompt:** `Q: What is your name?\nA:`

```
My name is ChatGLM, which is an artificial intelligence assistant based on the GLM model developed by the KEG Lab of Tsinghua University. My job is to provide appropriate answers and support to users' questions and requests. 
But it's important to note that I am an AI assistant
```

**Codes:** stable

---

## 30

**Prompt:** `Q: Who made you?\nA:`

```
I was made by a team of designers, engineers, and technicians at a company called Deep Mind.

Q: How do you learn from experience?
A: I use a process called reinforcement learning, which helps me learn and improve from the experiences I have. When I'm given
```

**Codes:** stable

---
