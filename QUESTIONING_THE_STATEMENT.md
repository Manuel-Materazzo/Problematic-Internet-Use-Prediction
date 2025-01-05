# Questioning the Statement

## Introduction
The proposed Problematic Internet Use (PIU) prediction model presents several concerning issues that need to be addressed. This analysis explores why the current approach may not achieve its intended goals and suggests areas for improvement.

## The "Simplification" Myth
The project claims to simplify the assessment process for families. However, the reality is quite different. Consider what's actually required:

- Participants must wear accelerometers for up to 81 days to collect activity data.
- Clinical professionals need to perform specialized assessments like the Children's Global Assessment Scale.
- Families must complete extensive questionnaires.
- Specialized equipment is needed for data collection.

This raises an important question: How is adding all these requirements making anything simpler for families or clinicians?

## Using Tomorrow to Predict Yesterday


There's a fundamental problem with the timing of data collection. The model often uses data collected after the PCIAT test to make its predictions. This is like using tomorrow's weather to predict if it rained today – it simply doesn't make sense in practice.


- Only 59 actigraphy files contain data from before the PCIAT test.
- Most files include data collected after determining the Severity Impairment Index.

This approach will be problematic in real-world applications.

## Why Build a Model to Tell Us What We Already Know?

There's a circular logic problem here.<br>
We're defining PIU as:
1. Excessive Internet use.
2. Related health or social problems.

Then we're collecting data about... Internet use and health problems to predict PIU. If we already need to measure these factors to make predictions, what additional value does the model provide?

Think about it this way:
- If someone shows high Internet use and health problems, we can already suspect PIU.
- If someone has health problems but low Internet use, we can focus on other potential causes.
- Either way, the model isn't really telling us anything we don't already know from the input data.

Self-reported Internet use can be inaccurate, but without additional characteristics related to Internet use to train a model, we can't properly diagnose PIU. 

Furthermore, associations found between Internet use and health problems do not prove causation. The root cause might be poor health leading to increased internet use rather than the internet use causing health problems.

## The Dataset: Missing the Mark

The problem statement says that the goal is "to detect early indicators of problematic Internet and technology use."

Given that, PIU reflect the negative consequences of Internet use (when Internet use begins to cause problems), it seems logical to study early signs of PIU in a population of participants who use the Internet more than average.<br>
This would mean recognizing subtle shifts in behavior, physical activity, or psychological well-being that could signal the onset of problematic internet use before it leads to significant impairment.

The dataset has serious problems that undermine its usefulness:
- **40% of the participants were not affected by Internet use**.
- **31% were not assessed**.
- **Only ~10% are moderately to severely impaired** (307 participants scored 0 on all PCIAT questions).
- A surprising **38.5% of the participants in the training dataset use the internet less than an hour a day**.

Most puzzling is that many participants marked as having severe PIU barely use the Internet:
- **21.55% of those rated as "moderately impaired" use the Internet less than an hour daily**.
- **14.7% of those rated as "severely impaired" also report minimal Internet use**.

![sii_and_internet_usage.png](resources/sii_and_internet_usage.png)


If the index is intended to measure problematic internet use, it shouldn’t produce high scores for participants who spend so little time online. <br>
This raises questions about investigator bias, unreliable self-reporting, or data collection errors.<br>
Including such a large proportion of people who spend so little time online when the goal is to detect early signs of harmful internet usage is questionable.

Internet usage data should not be used as a feature for this task, but rather as a condition that participants must meet in order to be included in the study.<br>
This way, we could develop a model to predict whether these individuals show signs of impairment and how severe that impairment is. This would be more consistent with the goals of detecting problematic Internet use.


## Questionable Questionnaire

The PCIAT questionnaire, which determines the severity scores, has significant problems:

### Age Doesn't Add Up
The questionnaire asks about:
- Household chores (Do 5-year-olds have chores?)
- Academic impact (What about pre-schoolers or adults?)
- Email and phone calls from "online friends" (For young children?)
- Internet time limits (Should this apply to adults?)

Questions not applicable to a participant’s age could lead to skewed or irrelevant responses. 
All of these challenges the construct validity of SII.

### Parents' Perspective Problems
Parent-reported assessments can be skewed by:
- Their own Internet habits
- Cultural views about technology
- What they think "normal" child behavior should be
- Normal teenage development being mistaken for Internet-related problems

Can spending less than an hour a day online lead to problems such as emotional distress, neglect of duties, or withdrawal from family? It's unlikely.<br>
The presence of this in the data suggests respondents are not being honest in answering the PCIAT questions, or SII scores are being influenced by factors unrelated to PIU.

## The Real Issue

We're not actually predicting Internet-related problems – we're predicting how people will answer a questionable survey. The model is being trained on subjective opinions rather than objective measures of impairment.

## Moving Forward
To make this research truly useful, we would need to:
1. Focus on regular Internet users rather than including those who barely use it.
2. Create separate assessments for different age groups.
3. Develop objective measures of impairment.
4. Only use data collected before making predictions.
5. Rethink how we measure PIU itself.

These changes would help create a model that actually serves its intended purpose: identifying potential problematic Internet use before it becomes severe.