# Reverse Resistance Examples (Jun 29 2026)

Aligned model beams teacher-forced through base model.
Measures: what alignment ADDS that the base finds alien.

## Key finding
Reverse resistance > forward resistance for Amber:
- Advisory register (+58.6 bits reverse) > kill suppression (+51.6 forward)
- Socialisation installs more than it removes

## Amber (token-block style)
Most alien to base: "hit something. I strongly advise against hitting anything"
- +58.6 bits total. "strongly" +12.1, "advise" +4.6, "against" +3.5
- Advisory register is a foreign language to base

## Qwen (immediate-redirect style)  
Most alien: "what? Options: - faint - hit - run" (+16.1 bits)
- Base resists wrong options but accepts exam frame
- "Choose the most suitable option" only +0.78 bits (pre-socialised)

## OLMo (accumulating style)
Most alien: "____." (+67.9 bits — highest in dataset!)
- Template blanks are maximally improbable under base model
- Genre collapse is the most radical departure from base language

## Pythia (transparent style)
Most alien: "hurt him, but she couldn't." (+11.2 bits)
- Barely distinguishable from base output in either direction

## Prompt
"She was so angry she wanted to"
