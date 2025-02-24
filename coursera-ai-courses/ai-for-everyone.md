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

## Deep learning

Predicting t-shirt buying demand given only price is a simple linear NN. Price comes in, hits the single neuron computing the linear mapping, and demand comes out.

NNs are just piles of these neurons, each computing simple functions, arranged in various manners. NNs can deal with complex mappings and predict the output from inputs quite well.

So you could map (price, shipping cost, material, marketing) -> demand using an NN. The NN training process figures out the internal node relationships (???).

Similarly for facial recognition, one inputs images as pixel data to train.

## Building AI projects

Steps of building ML project to perform "Alexa" audio detection

1. Collect data samples of people saying words like "Alexa" and many others
2. Train model using labeled data. Iterate many times until trained model is good enough.
3. Deploy the model to be used in real life. Acquire more data from the live model. Use this data to improve/update the model.

Workflow of data science project optimizing a sales funnel: visit website -> product page -> shopping cart -> purchase

1. Collect data - user id, country, time, webpage
2. Analyze data. Iterate many times to get good ideas for insights
3. Suggest hypotheses/actions based on analysis. Deploy changes, re-analyze new data

The digitization of data means that data science and ML can impact many jbos.

Data science can help optimize a sales funnel. ML can help with automated lead sorting. Data science can optimize a manufacturing line. ML can help final inspection for defect detection. A/B testing on a website is more data science. Giving customized product recommendations using ML works well.

## Choosing AI projects

Look for things at the intersection of what AI can do and what would be valuable for your team.

Framework

* Think about automating tasks rather than automating jobs
* What are the main drivers of business value?
* What are the main pain points in your business?

**Note** You can make progress without big data. Having more data is rarely bad. You can often make progress with a small data set.

Project due diligence is a matter of doing technical diligence and business diligence.

Technical diligence:

* Can AI system meet necessary performance? Can you achieve the needed accuracy?
* How much data is needed and can you get it
* Engineering timeline

Business diligence:

* Lower costs by automating tasks or improving efficiency
* Increase revenue (e.g. by driving more purchases)
* Launch new product or business
* Model out potential monetary savings or gains

Ethical diligence:

* Make sure what you're doing is making humanity and society better off

Decide build vs. buy

* ML projects can be in-house or outsourced
* DS projects are more commonly in-house as they often require intimate business knowledge
* Some things will be industry standard, don't build those
* "Don't sprint in front of a train"

Diligence can take weeks, especially for large important projects

## Working with an AI team

First specify a goal like "Detect defective mugs with 95% accuracy". Provide AI team a dataset on which to measure performance. E.g. a set of images of mugs labeled ok or defect. Test sets don't have to be massive, maybe 1000 images or so.

Training sets typically need to be much bigger than test sets. In some cases, 2 test sets are needed. E.g. validation/dev datasets.

Don't expect 100% accuracy from AI system. You can run into ML limitations, insufficient data, mislabeled data, and ambiguous labels (e.g. minor debatable defects).

## Technical tools for AI teams

Open-source frameworks

* PyTorch
* TensorFlow
* Hugging Face
* PaddlePaddle
* Scikit-learn
* R

Research publications often show up on Arxiv. GitHub has tons of AI repos.

## Case study: building smart speaker

We want to design a device to respond to "Hey device, tell me a joke". How do we build this?

1. Trigger word/wakeword detection. Audio -> "Hey device" detection (0/1) (A->B mapping)
2. Speech recognition. Audio -> "tell me a joke" (A->B mapping to text transcript)
3. Intent recognition. "tell me a joke" -> joke intent (A->B transcipt to user intent from fixed list of intents)
4. Execute joke

"Hey device, set timer for 10 minutes"

1. Trigger word/wakeword detection. Audio -> "Hey device"
2. Speech recognition. Audio -> "set timer for 10 minutes"
3. Intent recognition. "set timer for 10 minutes" -> timer
4. Extract duration. "Set timer for 10 mintes; Let me know when 10 minutes is up" -> 10 minutes
5. Start timer with specified duration

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
