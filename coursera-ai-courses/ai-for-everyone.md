# AI for Everyone

https://www.coursera.org/learn/ai-for-everyone

## Machine learning

Supervised learning learns mappings of A->B (input to output)

LLMs are built by using supervised learning (A->B) to repeatedly predict subsequent words. E.g. My favorite drink is lychee bubble tea.

Input: My favorite drink; Output: is

Input: My favorite drink is; Output: lychee

Input: My favorite drink is lychee; Output: bubble

Input: My favorite drink is lychee bubble; Output: tea

Input: My favorite drink is lychee bubble tea; Output: NA

When we train a very large AI system on a lot of data (100s billions of words or more) we get an LLM like ChatGPT.

Supervised learning took off recently in the 2010s because of the increase in 2 variables. First, tons of digital data are now available in many fields. Second, newer models have better performance with that increasing daata.

Specifically traditional AI flattens off quickly. Small neural nets continue to improve with a little more data. Medium NNs improve with even more data. Large NNs continue to improve.

## What is data

Input and output data can really be anything. For example pairs (size of house, # bedrooms) -> price or image -> cat? (0/1)

* Manual labeling is a way to get data. E.g. having a human label if images are or are not cats.
* Behavior observation can generate data. E.g. on an e-commerce site (user ID, time, price ($)) -> purchased (0/1) or (machine id, temperature C, pressure) -> machine fault
* Download from websites offering open datasets. There are tons across domains.

**Note** Don't build a huge data collection system without having the data analytics team in the loop from the beginning doing real work. They'll guide the development of the data sets.

**Note** Don't throw data at an AI/analytics team and assume it will be valuable.

**Data is messy** Data will have incorrect or missing values. Your data team needs to figure out how to clean and handle these things.

## Andrew's AI company transformation strategy

1. Use pilot projects to gain momentum
2. Build in-house AI team
3. Provide broad AI training to the whole company
4. Develop AI strategy
5. Develop internal and external comms

## Machine learning capabilities and inabilities

It is important to perform a technical feasibility study before embarking on a project. Such studies can take weeks even for experts like Andrew Ng.

"Anything you can do with 1 second of thought, we can probably now or soon automate"

What makes an ML problem easier? Learning a "simple" concept (e.g. less than 1 sec of thought). Lots of data available.

Examples

1. Can do: Vision systems identifying vehicle locations
2. Can't do: Identifying hand gestures (e.e. hands outheld saying stop, hitchhiking, left turn bike signal). These are also safety-critical so the systems have to be exceptionally accurate
3. Can do: X-ray diagnosis of pneumonia from 10,000 labeled images
4. Can't do: Diagnose pneumonia from 10 images and a medical textbook description. Humans could learn with this info

ML tends to work poorly learning from small amounts of data or when operating on new types of data. E.g. images are rotated or have defects when taken from another machine.

## Glossary

* **ANI** Artificial narrow intelligense like smart speaker, self-driving car, web search
* **Generative AI** Generative artificial intelligence like ChatGPT, Bard, Midjourney, DALL-E
* **AGI** Artificial general intelligence. Anything a human can do and more. Andrew says this is decades off. I agree.
* **Machine learning**
* **LLM** Large language model uses supervised learning (A->B) to predict the next word
* **Supervised learning** Learns a mapping of A -> B like email -> spam (0/1); audio -> text transcripts; English -> Chinese; ad, user info -> click (0/1); image, radar info -> position of cars, obstacles, etc.
* **Unstructured data** Generally things that humans produce without a direct scheme like images, audio, text.
* **Machine learning** Field of study that gives computers the ability to learn without being explicitly programmed (Arthur Samuel 1959 made chekers AI). Often results in building a system that can take inputs at any time and give outputs.
* **Data science** Science of extracting knowledge and insights from data. Often results in reports and presentations.
* **Deep learning** Neural networks take an input A and produce an output B which is the prediction. E.g. for houses (size, # bedrooms, # baths, is rennovated) -> price. The NN is a big equation producing the output.
* **


AI serves to impact most fields
