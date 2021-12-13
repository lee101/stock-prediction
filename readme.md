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
try trading strategy within bounds of the day predictions
