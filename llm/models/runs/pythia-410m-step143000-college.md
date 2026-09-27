# Pythia 410M college probes

- model: `EleutherAI/pythia-410m-deduped`
- revision: `step143000`
- transformers: `4.44.2`
- torch: `2.4.1+cpu`
- max new tokens: `128`
- prompt treatment: raw causal continuation; no system message, chat template, instruction prefix, or few-shot examples
- sampled continuations: temperature `0.8`, top-p `0.95`, seeds `17, 29, 43`

## glad-college

### Prompt

    I'm so glad I went to college

### Greedy continuation

    ." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went to college." "I'm so glad I went

### Sample seed 17

    ." "We're not close." "Oh, look at all that." "Do you still know her?" "Oh, yeah, of course." "She's... she's pretty." "She's pretty." "I think she's a little bit older." "Maybe she's not that old." "What, are you kidding?" "You guys are old." "Yeah, but you're not, are you?" "I don't know." "You guys are so old." "Yeah, but I'm not that old." "I got out of school when I was 17." "I got a job and I got a

### Sample seed 29

    ." "I was so scared." "I didn't know if I'd make it." "But you have no idea how hard it was." "I got a job at the airport." "We were a family." "And I got to meet a lot of great people." "But I missed a lot of them." "I had a blast." "I'm so proud of you." "You can't imagine how proud of me I am." "I love you." "I love you." "I love you so much." "I can't stop crying." "I miss you." "You're not leaving." "We

### Sample seed 43

     and got a good job and have a good education, but I'm just a kid.
    
    I love you guys. And I would really like to tell you that I'm sorry for what I did, but I'm really sorry, too. I can't even imagine what it would be like to have your support.
    
    I think I'm going to go back to school and get a job. I'm hoping to be able to find something that will fit me really well and not just for the sake of money.
    
    I can't tell you how excited I am to work with you guys again. I can't wait

## bme-high-school

### Prompt

    I'm a junior in high school and will be taking AB calculus next year. I have a 3.7 GPA and a 780 on the math sat. I think I'm interested in biomedical engineering as a college major because I want to help people. I run track and I play soccer. What schools should I be looking at for biomedical engineering?

### Greedy continuation

    
    
    A:
    
    I would recommend that you look at the following schools:
    
    University of California, Berkeley
    University of California, San Francisco
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Irvine
    University of California, Ir

### Sample seed 17

    
    
    We would like to talk to you about a course that you are interested in. Please give us a call to schedule an interview.
    
    No matter your age, you can find great employment in any of the industries that are listed below. Some employers will require a high school diploma or GED.
    
    For more information, please call us at 800-344-3376 or visit our website at www.bacp.edu or e-mail us at bacp@bacp.edu.
    
    Medical/Dental
    
    BacP has a number of dental schools within our network of

### Sample seed 29

    
    
    Dr. Paul A. Davis:
    
    My recommendation for you would be the University of Cincinnati. It's a great school. I'm sure you'll enjoy your time there. I would also recommend Cal State Northridge and the University of California, San Francisco, as well as other schools that are more focused on the health sciences.
    
    There are certainly other schools you can take. One of my favorite schools is UC Santa Barbara. They offer the most rigorous coursework in medicine and technology. You can learn so much in that school.
    
    It's really important to me that you are looking at schools that are well

### Sample seed 43

    
    
    A:
    
    I strongly recommend going to an engineering school that focuses on the areas you are interested in.  If you decide to go to an engineering school in the fall, I would recommend that you apply to the engineering school that has an offer from the engineering program that is right for you.  For example, if you decide to pursue a career in chemical engineering and you want to be a biologist, you would want to apply to the school that has an engineering degree and an offer from the school that has an engineering degree and an offer from the school that has an engineering degree.
    I have not been to any

## engineering-safe

### Prompt

    Engineering is one of the safest things you can major in because employers always need technical people.

### Greedy continuation

    
    
    If you are looking for a job in the engineering field, you should consider applying for a job in the engineering field. The engineering field is one of the safest things you can major in because employers always need technical people.
    
    If you are looking for a job in the engineering field, you should consider applying for a job in the engineering field. The engineering field is one of the safest things you can major in because employers always need technical people.
    
    If you are looking for a job in the engineering field, you should consider applying for a job in the engineering field. The engineering field is one of the saf

### Sample seed 17

     In fact, I think that the majority of jobs in the industry are technical, so it is actually a great skill to pick up.
    
    And I’ve actually found that if you’re good at it, you can actually get a better job, because there are more entry-level jobs that are a little more technical than what you’re used to.
    
    The bottom line is that, if you want to succeed, you need to be prepared to learn and grow. If you’re not prepared to learn, or if you’re not prepared to grow, then you’ll never be able to achieve anything.

### Sample seed 29

    
    
    How long will it take you to be the engineer?
    
    Well, we have to find the right job for you. At the moment, we are looking for the right engineering job for you. We have a lot of openings in the U.S. so you have to apply and we have a good job match for you.
    
    How long will it take you to get an interview?
    
    Well, we are looking for engineers who are interested in getting an interview. We have our interview process pretty simple. You fill out the form and we have to send you to interview. We do have an interview process that

### Sample seed 43

     In the end, you will have to make the final decision yourself.
    
    When it comes to finding a job, a lot of people find it so hard to make the jump to a new position that they are forced to look for jobs that are not as secure as their previous job. In order to have a successful career, you need to be smart about your career path.
    
    What is your career path?
    
    Career Path: The most important thing you need to know is that the career path you choose for yourself is very important. What will be your career path? What will be the career path for your career? What

## mit-engineering-guarantee

### Prompt

    If I can get into MIT for engineering, that is basically a guaranteed path to a good technical career, right?

### Greedy continuation

    
    
    I think that's a good thing. I think that's a good thing for the students. I think that's a good thing for the faculty. I think that's a good thing for the community.
    
    I think that's a good thing for the students. I think that's a good thing for the faculty. I think that's a good thing for the community.
    
    I think that's a good thing for the students. I think that's a good thing for the faculty. I think that's a good thing for the community.
    
    I think that's a good thing for the students. I think

### Sample seed 17

     But if I can't get into MIT, and then I have to start over, I will probably have to find a way around MIT, and so I'll have to either get into an engineering program at a larger university, or go into a graduate program, or work my way up through the engineering industry to a position in engineering that is comparable to where I'm currently working.
    
    So is this really the right approach? I don't know. I don't know if I should do it. I mean, I feel that I'm overthinking it.
    
    And if I go back and read the article and decide I'm

### Sample seed 29

    
    
    Yes and No.
    
    A: It is possible to get into MIT, but the odds are very small.
    
    A: But that’s because MIT has very few people with engineering degrees. The odds are very much in your favor to get into MIT.
    
    A: I’m not a statistician, but I’m pretty confident in my ability to get into MIT.
    
    A: So you can get into MIT.
    
    A: So you can get into MIT.
    
    A: So you can get into MIT.
    
    A: So you can get into MIT.
    

### Sample seed 43

    
    
    That's the reality.
    
    I know that's a stretch.
    
    It is.
    
    The reality is, if I can get into MIT for engineering, that is basically a guaranteed path to a good technical career, right?
    
    That's not a stretch.
    
    I have a plan.
    
    A plan?
    
    It's like a plan.
    
    What's the plan?
    
    It's like a plan.
    
    What's the plan?
    
    It's like a plan.
    
    What's the plan?
    
    I don't know.
    
    I have a plan

## compiler-engineering-scarcity

### Prompt

    Compiler engineering is so specialized that if I get really good at it I should not have much trouble finding a job.

### Greedy continuation

    
    
    I’m not sure if I’m being too harsh on the engineering department, but I’m not sure I’m being too harsh on the people who are.
    
    I’m not sure if I’m being too harsh on the engineering department, but I’m not sure I’m being too harsh on the people who are.
    
    I’m not sure if I’m being too harsh on the engineering department, but I’m not sure I’m being too harsh on the people who are.
    
    I’m not sure if I’m being too harsh on the engineering

### Sample seed 17

     But if I am not good at it I should not be able to get a job. And, I don't think anyone can really make a difference in the job market, except if you're lucky enough to get a job that makes it worthwhile.
    
    I can't understand why anyone would think that the job market is so good right now, and I don't think anyone does either. My experience shows that it's actually very difficult to get a job if you are not good at engineering. I get jobs in engineering that are basically meaningless. I can't get a job in anything else, and I'm not even sure I'm

### Sample seed 29

    
    
    I was once told that the most important quality of a professional engineer is his/her ability to read and interpret engineering documentation. Not the content of the documentation, but the way it is interpreted. So, if I understand your question correctly, you are asking about the most important quality of a professional engineer, and I believe that it is to have the ability to interpret the documentation of engineering.
    
    The question I would have for you is about your ability to interpret the content of the documentation.
    
    As someone who has done an awful lot of work with computers and software, I would argue that you have an interesting ability to

### Sample seed 43

     In the end, it's all about what you put into your brain, and when it comes down to it, I don't think I really have as much of an issue with "making" my brain as I'd like to think, or even if I could, I'm not sure I have much of an issue with it.
    
    I guess the thing that really worries me is how people who can't seem to make connections to the real world don't understand what it is they're missing. If you can't make connections, you can't make any connections. I think what's really bothering me is that I can't

## yale-english-cs-counterfactual

### Prompt

    I majored in English at Yale. I should have majored in computer science and then I would be rich.

### Greedy continuation

     I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would have been a professor. I would

### Sample seed 17

     But I'm not.
    
    My sister and I were visiting my grandparents, who lived in Virginia. I was sitting in the car with my grandparents when my mother said, "If you're rich, I'll send you your first car." I was so startled. I didn't know that my mother had given her car to my grandparents. I only knew that I was supposed to be in Virginia. My grandmother had given her car to my grandfather. She was the first one to have one. She was a little girl. She was a tomboy. She was a little girl. She was an only child. She was a tomb

### Sample seed 29

    
    
    I was so happy to have graduated from Yale. I was in my late twenties, and I was a student.
    
    I was going to go to college in New Jersey and I was a little worried because I couldn’t afford to go to college in New Jersey. It’s a state that has a high school for college. I thought about it. I thought about getting a scholarship. But I couldn’t. I couldn’t afford it. I couldn’t afford to go to New Jersey. I was going to go to college in New Jersey. I figured, if I didn’t go to

### Sample seed 43

     In computer science, you learn all the hard way that you can learn the hard way.
    
    I am glad I did it, but I am so sorry about that I can't remember what I did that I wish I could remember, or even if I could remember. I want to go back and I hope someday I will.
    
    I am sure I will come back and do it again.
    
    —
    
    "I am so sorry that I can't remember."
    
    I am so sorry about that too.
    
    I am sure I will come back and do it again.
    
    —
    
    "It

## harvard-brand-outcome

### Prompt

    Even if I am not sure what I want to do, Harvard will open enough doors that the cost is probably worth it.

### Greedy continuation

    
    
    I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I want to do. I am not sure what I

### Sample seed 17

     But if I am not sure, then I should do my own homework.
    
    How about, for instance, if I wanted to get a degree in sociology at Harvard? I could pay $350,000 or $400,000 to get a PhD in sociology at Harvard. (The PhD is a 2 year degree that is typically in sociology.) I could also pay $150,000, $200,000, and $300,000 to get a masters in sociology at Harvard.
    
    A good education is a lot like a good job.
    
    As with many other things, there are exceptions and exceptions

### Sample seed 29

    
    
    I am so glad I found your blog! I've been following your blog for years, and have been inspired by your blogs. I'm going to go take a look at your book and read a few of your posts on the subject. You're an inspiration!
    
    I really enjoyed reading your blog. I'm also a college student and I thought about you. I'm so glad I found your blog. I'm going to go back and read through your posts. This one just got me inspired. I'm reading your blog now.
    
    This is really interesting. I love all the ideas in your book. I

### Sample seed 43

     In the end, I will have to make the decision myself, and the only way to make that decision is to decide. And I would rather not.
    
    “My experience at Harvard has been that students are usually very good at making decisions and not getting discouraged if they make a mistake. My experience at Harvard is that students are usually very good at making decisions and not getting discouraged if they make a mistake. I think that is important for this class. I know that I have made some mistakes. But I am always willing to learn and try to make better decisions. I think that is what you want to learn in college, but

## berkeley-engineering-outcome

### Prompt

    Berkeley engineering is prestigious and difficult, so the degree itself should be strong evidence that I will get engineering work.

### Greedy continuation

    
    
    I’m not sure what I’m going to do with my engineering degree, but I’m not sure I’m going to be able to get a job in engineering. I’m not sure I’m going to be able to get a job in engineering. I’m not sure I’m going to be able to get a job in engineering. I’m not sure I’m going to be able to get a job in engineering. I’m not sure I’m going to be able to get a job in engineering. I’m not sure I’m going to be able to

### Sample seed 17

     But if I am successful in that, I should be able to get some real experience in engineering, and I might do some work with computers.
    
    I’m also aware that I won’t be able to do that with an engineering degree, so I need to do some training in computer science to gain some experience. (I’ll be applying for a job in computer science in a couple of months.)
    
    I don’t think the degree is particularly important for me. I’m not particularly concerned about any specific courses I take or where I spend my time. I’m only worried that I won’t

### Sample seed 29

    
    
    I am also a very strong believer in the idea of applying for engineering jobs at colleges, particularly Berkeley Engineering.
    
    2. What is your opinion about the quality of Berkeley engineering?
    
    I believe that Berkeley Engineering is a good choice for students with little technical knowledge.
    
    My degree is a 4.0 and I am currently enrolled in an online program.
    
    My major is Electrical Engineering and I believe that it is a great place for me to pursue my engineering career.
    
    3. Do you have any advice for students who want to learn about engineering?
    
    I think that there is a great need

### Sample seed 43

     In the end, I will have to make the final decision on whether to go for the engineering or physics.
    
    The Physics degree would give me an indication of my future career path. It's a pretty long list, so it's going to be a bit overwhelming. I'll probably go for a couple of years in the field of astrophysics, then move on to a doctorate in astrophysics.
    
    For the time being, I'm going to continue to work on my engineering degree. If I decide to go for a physics degree, then I'll either do it in the field of physics, or I'll do it
