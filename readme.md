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

total_profit avg per symbol:
0.013155548607755

from trying to match percent change
val_loss: 0.0251106689684093
====
val_loss: 0.024709051416721195
total_buy_val_loss: -0.006730597996011056 < - losses at end of training/overfit
total_profit avg per symbol: 0.013266819747514091


===removed clamping in training - slightly better
val_loss: 0.024133487895596772
total_buy_val_loss: -0.0067360673833718465
total_profit avg per symbol: 0.013524375013730605


=====torchforecastiong
mean val loss:$0.04344227537512779
val_loss: 0.031683046370744705

again 30epoc
val_loss: .03192209452390671

0.03335287271 avg profit trading on preds is high though


{'gradient_clip_val': 0.021436335688506693, 'hidden_size': 100, 'dropout': 0.13881629517612382, 'hidden_continuous_size': 61, 'attention_head_size': 3, 'learning_rate': 0.0277579953131985}
mean val loss:$0.02416972815990448
val_loss: 0.031672656536102295
total_buy_val_loss: 0.0
total_profit avg per symbol: 0.0

Process finished with exit code 0
=========

current day Dec18th
Best val loss: -0.0037966917734593153
Best current profit: 0.0037966917734593153
val_loss: 0.03043694794178009
total_buy_val_loss: 0.009012913603025178
total_profit avg per symbol: 0.0021874699159525335
========== running after htune:

running Training time: 0:00:01.827697 Best val loss: -0.00021820170513819903 Best current profit: 0.00021820170513819903
val_loss: 0.03161906823515892 total_buy_val_loss: -0.0067360673833718465 total_profit avg per symbol:
0.013325717154884842

Process finished with exit code 0



=======
take profit training

Training time: 0:00:01.391649
Best val loss: -0.0008918015519157052
Best current profit: 0.0008918015519157052
val_loss: 0.0
total_buy_val_loss: 0.0018733083804060395
total_profit avg per symbol: -0.0018733083804060395
'do_forecasting' ((), {}) 44.71 sec
===== all bots

Training time: 0:00:01.933525
Best val loss: -0.008965459652245045
Best current profit: 0.008965459652245045
val_loss: 0.029988354071974754
total_buy_val_loss: 0.008610340521651475
total_profit avg per symbol: 0.004202203740229986
'do_forecasting' ((), {}) 302.33 sec
