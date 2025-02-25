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

## Case study: self-driving car

Steps for deciding how to drive

1. Gather input from image, radar, lidar, GPS, accelerometers, maps
2. Detect other cars, pedestrians, obstacles, etc. (supervised ML using A->B mapping sensing to bounding boxes/regions)
3. Trajectory prediction of obstacles
4. Lane detection
5. Traffic light, stop sign, etc. detection
6. Motion/path planning
7. Translate into steering, acceleration, braking

## Example roles on AI team

Regardless of a team size there are common roles and responsibilities which show up on AI teams

* **Software engineers** write software to perform actions, build ML algorithms, etc.
* **Machine learning engineers** write software to build ML algorithms, gather data, train networks, test networks
* **Machine learning researcher** extends state-of-the-art in ML
* **Applied ML scientist** blends the roles of ML engineers and ML researchers by staying up to date on the state of the art and adapting to current problems
* **Data scientists** examine data, provide insights, present to teams and executives, sometimes work on ML
* **Data engineers** organize data to ensure it's saved in an accessible, secure, cost-effective manner
* **AI product managers** help decide what to build, what's feasible, what's valuable

## AI transformation playbook

Read [AI-Transformation-Playbook.pdf](./AI-Transformation-Playbook.pdf). Download from [here](https://landing.ai/case-studies/ai-transformation-playbook).

Andrew Ng wrote an "AI Transformation Playbook" for companies to become great at AI. The steps are

1. Execute pilot projects to gain momentum
2. Build an in-house AI team
3. Provide broad AI training
4. Develop an AI strategy
5. Develop internal and external communications

### 1. Execute pilot projects to gain momentum

Success of pilot projects is more important than project value. Success shows other teams what things are possible. Just get the flywheel turning.

Show traction quickly (6-12 months). Pilot projects can be in-house or outsourced. Outsourcing can help build momentum faster.

### 2. Build an in-house AI team

Andrew suggests building a centralized AI team and then integrating AI team members into other business units throughout the company as needed.

The centralized AI team allows that team to build a community of like-minded people who can then service the company. The AI team can also build company-wide AI platforms to service the entire company.

AI team should be under a director like CTO, CIO, CDO, or CAIO. The CEO should provide initial funding for the AI team to kickstart things.

### 3. Provide broad AI training

Multiple people at multiple levels need to understand AI

* Executives and senior business leaders should learn what AI can do for the business, AI strategy, and resource allocation.
* Leaders of divisions working on AI projects need to set project directions (technical and business diligence), resource allocation, and monitor progress.
* AI engineer trainees. Existing engineering staff can be trained (100 hrs ish) to build and ship AI software, gather data, and execute on specific AI projects

A smart CLO should curate content rather than create content. There's tons of info in online courses, books, videos, and blog posts that already exist. Use these rather than creating whenver necessary.

### 4. Develop an AI strategy

Trying to define AI strategy prior to getting experience with steps 1-3 can be premature. It's hard to know what is possible and valuable without experience and can lead to poor strategies beforehand.

Design strategy aligned with the "Virtuous cycle of AI": better product -> more users -> more data -> AI system -> better product. This shows why existing players often have huge advantage in fields like web search.

For example, Blue River used smart phones to take pictures of crops and weeds to train an ML system to identify weeds. They built an underwhelming but functional product they could demo. Then they could deploy the system, gather more data, improve the product, and iterate.

Consider creating an AI data strategy.

Think of strategic data acquisition. E.g. create a free service meant to collect data. E.g. free email or photo service to mine tons of data.

Build a unified data warehouse so that the whole company can pull in all necessary data available to the company.

Create network effects and platform advantages. AI can be an accelerator in industries with winner take all dynamics (e.g. Uber & Lyft, or social media). Gaining a large share can lead to accelerating growth.

Strategy is very company, industry, and situation-specific.

AI can fit into more traditional frameworks. E.g. AI can reduce costs or increase value for traditional low cost/high value frameworks.

**Important** AI is a superpower. Build businesses that make humanity better.

### 5. Develop internal and external communications

AI can change a company and products. You need to communicate about it:

* Investor relations
* Government relations. E.g. in healthcare showing how AI can be used in ways that protect patients
* Customer/user education as your products change
* Talent/recruitment. Showing success and strategy gives talent a reason to join your company
* Internal communication as company strategy changes

## AI pitfalls to avoid

Don't

* Expect AI to solve everything
* Hire 2-3 ML engineers and solely rely on them to come up with use cases or do everything
* Expect AI projects to work the first time
* Expect traditional planning processes to apply without modifications
* Think you need superstar AI engineers before you can make any progress

Do

* Be realistic about what AI can do given the limitations of technology, data, and engineering resources
* Pair engineering talent with cross-functional teams to find feasible and valuable projects. Use scarce ML engineering resources carefully to augment your organization.
* Plan for AI development to be iterative with multiple attempts needed
* Work with AI teams to establish timelines, milestones, KPIs, etc. These differ for AI projects. Leverage experienced AI engineers to help build these.
* Just get started

## Taking your first steps in AI

* Get friends to learn about AI
    * Take courses, doing a reading group, etc.
* Start brainstorming projects
    * No project is too small. Starting small and succeeding is better than going big and failing.
* Hire a few ML/DS people to help out
* Hire or appoint an AI leader (VP of AI, CAIO, etc.). Can hire folks without a senior AI leader.
* Discuss AI possibilities with CEO/board
    * Will your company be more valuable or effective if it were good at AI?

## Survey of major AI application areas

* Computer vision
    * Image classification & object recognition
        * Face recognition
    * Object detection. Find presence and position of objects in image.
    * Image segmentation classifies each pixel as being part of some object or not. E.g. these are pedestrians, cars, etc.
    * Tracking follows where objects move over time in a video
* Natural language processing (deep learning specifically)
    * Text classification: email -> spam/not spam or product description -> product category
        * Sentiment recognition takes user text and computes a user sentiment, review level, etc.
    * Information retrieval like web search
    * Name entity recognition finds names, locations, companies, phone numbers, etc. in text
    * Machine translation
* Speech
    * Speech recognition converts digital audio to text
    * Trigger word/wakeword detection finds certain words like "Alexa", "Hey Google"
    * Speaker ID identifies speakers based on listening to speech
    * Speech synthesis or text to speech (TTS) takes input text and produces speech for it
* Generative AI
    * AI systems that can produce high-quality content like text, images, and audio
    * LLMs are great at text generation, editing, generating summaries, chatting, etc. LLMs can help increase productivity via brainstorming, learning, etc.
    * Image generation from text descriptions
    * Audio generation can generate speech, music, or sound effects from a prompt like "drum solo 140 bpm"
* Robotics
    * Perception: understanding the world around a machine
    * Motion planning helps find a path for the robot to follow
    * Control sends commands to motors to execute actions smoothly
* General ML
    * Unstructured data like images, audio, text is popularized in the media compared to structured data
    * AI on structured data creates tremendous data even though it's less commonly reported on

## Major AI techiques

### Unsupervised learning

Doesn't provide the set of desired labels or categories

Clustering

Consider a grocery selling packets of potato chips based on cost per packet. A clustering algorithm would group data (# packets bought, price per packet) into various clusters of shoppers.

### Transfer learning

Suppose you have a car detection algorithm and need to apply this to golf cart detection. Transfer learning helps you learn from task A to allow you to work on task B with much smaller data sets.

Many computer vision systems are built using transfer learning

### Reinforcement learning

Suppose you want to teach a helicopter to fly itself. Supervised learning is not effective as it's hard to define the outputs.

You let the system behave as it will encouraging good behaviors and discouraging bad behaviors. It uses a "reward signal" to tell the AI when it is doing well or poorly. The training system automatically learns to maximize rewards.

Reinforcement learning has been applied well to robots, autonomous vehicles, or games like chess or go.

Reinforcement learning requires many many experiments to learn. Systems where simulation are present are best for reinforcement learning.

### Generative adversarial networks (GANs)

Synthesize new images from scratch

### Knowledge graph

Collection of related knowledge on a given topic?? He didn't really go into this very well.

## Realistic view of AI

* Too optimistic: AI will solve all of humanity's problems creating a utopia
* Too pessimistic: Super intelligent/sentient killer AI can lead to human extinction
* Accurate: AI is a powerful tool with limitations. We can control potential harms and create tremendous value.

Limitations of AI

* Performance limitations
* Explainability is hard but sometimes doable
* Biased AI through biased data (e.g. gender, ethnicity, etc.). An AI system can easily come to discriminate based on these variables.
* Adversarial attacks on AI, especially on those making important economic decisions

## Glossary

* **ANI** Artificial narrow intelligence like smart speaker, self-driving car, web search
* **Generative AI** Generative artificial intelligence like ChatGPT, Bard, Midjourney, DALL-E
* **AGI** Artificial general intelligence. Anything a human can do and more. Andrew says this is decades off. I agree.
* **Machine learning**
* **LLM** Large language model uses supervised learning (A->B) to predict the next word
* **Supervised learning** Learns a mapping of A -> B like email -> spam (0/1); audio -> text transcripts; English -> Chinese; ad, user info -> click (0/1); image, radar info -> position of cars, obstacles, etc.
* **Unstructured data** Generally things that humans produce without a direct scheme like images, audio, text.
* **Machine learning** Field of study that gives computers the ability to learn without being explicitly programmed (Arthur Samuel 1959 made chekers AI). Often results in building a system that can take inputs at any time and give outputs.
* **Data science** Science of extracting knowledge and insights from data. Often results in reports and presentations.
* **Deep learning** Neural networks take an input A and produce an output B which is the prediction. E.g. for houses (size, # bedrooms, # baths, is rennovated) -> price. The NN is a big equation producing the output.

AI serves to impact most fields
