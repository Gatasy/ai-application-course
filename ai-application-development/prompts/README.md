# Prompt Module

## Files

router.txt
Question routing prompt

disease.txt
Disease QA prompt

drug.txt
Drug QA prompt

exam.txt
Exam QA prompt

## Placeholder

Disease/Drug/Exam prompts require:

{context}
{question}

Router prompt requires:

{question}

## Expected Router Output

{
  "category":"Disease"
}

{
  "category":"Drug"
}

{
  "category":"Exam"
}