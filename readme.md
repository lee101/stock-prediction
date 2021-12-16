npm install -g selenium-side-runner
npm install -g chromedriver   


unceirt + predicted next? - seems bad
2.8 high val loss when ran on high stocks, volatility bonus of 1 made profit

on smaller stocks:
val_loss: 2.2425003216734956

important to constrain to stocks you think are good
10% up but lost a lot on unity 

fewer stocks -> 10%

        'GOOG',
        'TSLA',
        'NVDA',
        'AAPL',
        # "GTLB", not quite enough daily data yet :(
        # "AMPL",
        "U",
        # "ADSK",
        # "RBLX",
        # "CRWD",
        "ADBE",
        "NET",

on more incl asx
val_loss: 0.29750736078004475
new val loss when having more data in sequences: 0.3078561797738075
just more history: 0.3317318992770236

flipped loss:
val_loss: 0.274111845449585

now with aug:
val_loss: 0.12366707782660212


## random augs:
+1000 epocs
total_profit avg per symbol: 0.047912802015032084
now:
 04841010911124093 
now 0.06202507019042969

total_profit avg per symbol: 0.0720802800995963 

after random aug + 1000epocs :

0.09813719136374337

leave it to train 100k
total_profit avg per symbol: 0.18346667289733887
graphs not looking good though..


now 67.57110960142953 ???


=== now we are training on better money loss/trading
Training time: 0:00:21.642027
Best val loss: -0.0022790967486798763
Best current profit: 0.0022790967486798763
val_loss: -0.010014724565727162
total_profit avg per symbol: 0.022031369014174906 <- daily


===== 15min data

val_loss: 2.8128517085081384e-06
total_profit avg per symbol: -8.676310565241302e-08
better hourly? try dropping 4?
==========
drop 1/2 1/2 not good either

val_loss: 1.0086527977039492e-05
total_profit avg per symbol: -3.3665687038109127e-07

===== passing also data in of high//low
Best current profit: 0.006474322639405727
val_loss: -0.024440492995630336
total_profit avg per symbol: 0.055027083498743634

total_profit avg per symbol: 0.05783164164083199



=====
try 15min data and shift results by 4hours or 1 day
try trading strategy within bounds of the day predictions+


===== dropout+relu
val_loss: -0.009048829903456124
total_profit avg per symbol: 0.03414255767188412

only relu even lower?
0.03064739210509515
only dropout?
0.046652720959281524

numlaryers 2->6
0.06964204791370121 wow!
training time 20-48

numlayers 32 1k epocs
0.0170769194062945 terrible 

numlayers 32 10k epocs
val_loss: 0.006968238504711621
total_profit avg per symbol: 0.02565125921381299

===todo predict output length of hodl
also predict percent away from market buy/sell, - compute open/close based trading sucucess loss

================= wow!!!
val_loss: 12.973313212394714
total_profit avg per symbol: 4.278735787607729


==== after fixing bug
Best current profit: 0.0022790967486798763
val_loss: -0.0019214446920077233
total_profit avg per symbol: 0.02520072289090347

Process finished with exit code 0



-===back to 6ch GRU

val_loss: -0.009624959769610086
total_profit avg per symbol: 0.014541518018852617

run for 10k epocs?
Best current profit: -1.7888361298901145e-06
val_loss: -0.006090741769895658
total_profit avg per symbol: 0.012417618472702507


lower loss
total_profit avg per symbol: 0.029944509490936373
========== percent change augmentation wow!
val_loss: -0.04609658126719296
total_profit avg per symbol: 0.0835958324605599

==== adding in open price
0.06239748735060857

====back down after changing the +1 loss function
val_loss: -0.004483513654122362
total_profit avg per symbol: 0.011341570208969642

now with added open price
val_loss: -0.00627030248142546
total_profit avg per symbol: 0.013123613936841139

-- scinet
0.01109844121


-- no scinet


total_profit avg per symbol: 0.013164439284082619


-- unlocking clamping in training + lower random aug

total_profit avg per symbol: 0.013164260120902327



====clamping grads after epoc
model per isntrument - saving first time saved

total_profit avg per symbol: 0.01066157597961137

-- omni saved model from prev non instrument

TODO global model?

===cancel model saving - bring back more augmentation
- clamping after 300epocs
- 
total_profit avg per symbol: 0.01328011912482907


==== everything is early stopping
turned off early stopping after 30 unsuccessful epocs

total_profit avg per symbol: 0.013270534476760076

====== softmax

total_profit avg per symbol: 0.009848967136349529

====== sigmoid is better?

0.006266821141252876
====

total_profit avg per symbol: 0.006266821199460537
