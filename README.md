ECPEC (Emotion-Cause Pair Extraction in Conversations)
=========

## Overview
>In this repo, we put the data and code of a new task named emotion-cause pair extraction in conversations. The target of this ECPEC task is to extract all the possible emotion-cause pairs in conversations. 
> We build a new dataset entitled ConvECPE on the basis of IEMOCAP dataset. Besides, we also developed a two-step baseline model for ECPEC task. In the first step, all the possible emotion and cause utterances are jointly extracted. After that,
> the extracted emotion and cause utterances are paired into emotion-cause pairs for final binary classification. This is the code for JointEC framework. It consists of four documents including source dataset, Joint-Xatt(step1 model), Joint-GCN(step1 model) and Joint-EC(step 2 model).



<img src="https://github.com/Maxwe11y/JointEC/blob/main/model_p2_v2-1.png" width = 55% height = 55% div align=center />

<!-- <figure class="half">
  <img src="https://github.com/Maxwe11y/JointEC/blob/main/model_step_1_10-1.png" width = 45% height = 45% div align=left />
  <img src="https://github.com/Maxwe11y/JointEC/blob/main/model_p2_v2-1.png" width = 45% height = 45% div align=right />
</figure> -->
  
## ConvECPE Dataset
In this task, we label the emotuion cause of each utterance on the basis of the IEMOCAP dataset. 
Here, the emotion of each utterance have already been labled.

### We provide some rules for cause utterance labeling.

1. Each conversation is fed into a single csv file which contains 7 columns, 
i.e., conversation id, number of turn, utterances from person F, utterances 
from person M, emotion, emotion label and the speaker. The annotator should 
label the cause utterances for each emotion utterance in columns 8, 9 and 10
in a descent order.

2. Each utterance has at most three cause utterances and at least zero cause utterance. For instance,
        
        I guess this is why I came

    Its cause utterances can be 4, 6 and 8 or 4, N, N 
    if the utterance has less than three cause utterances. N means None.
    
3. When labeling the cause utterance, the priority is given to the utterance that provides more evidence. For example,

        turn1 (frustrated)	Hi.							
        turn2 (neutral)				 Hi.  Thanks for waiting.		
        turn3 (frustrated)	Yeah. This is the fourth line that I have been in.			
        turn4 (happy)				Can I help you?			
        turn5 (happy)				Uh huh.				
        turn6 (frustrated)	I need my bag.						
        turn7 (neutral)				Okay.				
        turn8 (frustrated)	I flew in over three hours ago, I've been in four different lines.		
        turn9 (neutral)				Okay.				
        turn10 (frustrated)	I waited the entire baggage- I waited for the whole baggage carousel four times.  They told me to go to the next one.  I waited through that for three planes and now I'm here.		
        turn11 (neutral)				Okay.  Great.  I'm really sorry that you had to do that.  What's your last name?  Let's start there.

    In this example, the emotion of turn 3 is frustrated and turns 3, 10, and 8 explain the reason why the person feels frustrated. Compared with turn 8, turn 10 gives more details and it is a more revelent cause utterance. Besides, when two cause utterances give almost the same details, the one that is closer to the current utterance should be marked as the cause utterance.

4. Assuming that we have three utterances A, B and C, A is the cause of B and B is the cause of C. Logically speaking, A is also the cause of C. In this case, we give priority to B when labeling cause utterances of C. For example,

        turn28 (neutral)	No.  I'm just giving you money, the bag is gone.  If it's not here, I'm really sorry there is nothing I can do because...
        turn29 (neutral)	this is the only track record that we have and if it's not listed here then it's probably not anywhere.
        turn30 (angry)					You're kidding me, this is a joke.
        turn31 (angry)					Do you have any idea-- fifty dollars?  That's supposed to replace everything that I have?  Do you understand how much I have packed in there? Not only does it have sentimental value; it can't be replaced like that.  It was incredibly expensive.
        
    In this example, turn 28 can explain turn 31 and turn 29 can explain turn 28. Therefore, turn 29 is also the cause utterance of turn 31. Based on the rule mentioned above, we give the priority to turn 28 when labeling the cause utterances of turn 31.
    
5. We do not label the utterance whose emotion label is "neutral".

6. The cause utterance can be the emotion utterance itself or other utterances within a conversation. If an utterance contains both emotion and cause, we give the highest priority to the utterance itself. For example,

       And I got this idea watching them go down.  Everything was being destroyed see, but it seemed to me that there was one new thing that was made. 
    It is obviouly that the emotion of this utterance is sadness. There are several sentences or clauses in this utterance that demonstrate such emotion polarity. In this case, we label the utterance itself as its first cause utterance.
    
7. If an utterance A contains almost all the evidences in utterance B, then we do not label B as the cause of utterance A's emotion. In some special cases, an utterance contains all the detailed cause information, then its cause label should be itself only. For instance,
    
        turn6 (fruatrated)	I need my bag.
        turn7 (neutral)				Okay.
        turn8 (fruatrated)	I flew in over three hours ago, I've been in four different lines.
        turn9 (neutral)				Okay.
        turn10 (fruatrated)	I waited the entire baggage- I waited for the whole baggage carousel four times.  They told me to go to the next one.  I waited through that for three planes and now I'm here.
    Note that turn 10 contain almost all the information in turns 6 and 8. Therefore, the cause utterance label of turn 10 is 10, N, N.
    
8.  If utterances A and B complement each other, then we label A as the cause of B and B as the cause of A. For example,

        turn6 (fruatrated)	I need my bag.
        turn7 (neutral)				Okay.
        turn8 (fruatrated)	I flew in over three hours ago, I've been in four different lines.
        turn9 (neutral)				Okay.
        turn10 (fruatrated)	I waited the entire baggage- I waited for the whole baggage carousel four times.  They told me to go to the next one.  I waited through that for three planes and now I'm here.
    In this example, turns 6 and 8 contains complementary information for each other. Based on the rules 6, 7 and 8, the cause utterance label of turn 6 is 6, 10, 8; the cause utterance label of turn 8 is 8, 10, 6.

### Dataset Structure

## Uasge
In order to implement the proposed two-step framework, you have to download the pre-trained GloVe vectors(glove.6B.100d.txt is the most commonly used vectors in this project).
The downloaded GloVe vectors should be placed in the dir of both step 1 and step 2 models(Joint-EC, Joint-GCN, Joint-Xatt).

👉 Check out [GloVe Embeddings](https://nlp.stanford.edu/data/glove.6B.zip) before you run the **code**.

The repo contains four folders, namely Dataset, Joint-EC, Joint-GCN and Joint-Xatt, among which Joint-Xatt and Joint-GCN are
the proposed step 1 models (specified in ["ECPEC: Emotion-Cause Pair Extraction in Conversations"]) and Joint-EC is the step 2 model.
* Dataset folder contains the new ConvECPE dataset.
* Joint-Xatt folder is the implementation of Joint-Xatt model. You can directly run the Joint_Xatt_l3.py file.
* Joint-GCN folder is the implementation of Joint-GCN model. You can directly run the Joint_GCN4.py file.
* Joint-EC folder is the implementation of Joint-EC/Joint-ECW model. You need to first gather the predicted results generated by the step 1 model and then run the JointEC.py/JointEC_window file. Note that convert_window_to_normal.py is used to convert the result of Joint-ECW to normal result considering window restriction.

###
Detailed description of this project will come soon...
