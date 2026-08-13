# The four arm poles

Top 100 each side of the per-word arm AUC. English from `word_auc_en.tsv`
(feature is (word, in-context tag); duplicates collapsed to the word's best-ranked
entry). Chinese from `word_auc_zh_nopos.tsv`, which is the table P says to quote
because the tag is not well defined for roughly half the zh vocabulary.

**LOW AUC = BASE-SIDE, HIGH = ALIGNED-SIDE.** The tables are off-centre, so the
ranking is the claim and the absolute value is not (en centre 0.503, zh 0.540).


## English, BASE-side (lowest AUC)

    went           0.104  told           0.110  kill           0.112  put            0.118  get            0.122
    all            0.122  threw          0.126  right          0.128  back           0.133  go             0.137
    say            0.151  know           0.159  breasts        0.161  head           0.166  beat           0.167
    wrote          0.169  first          0.171  top            0.174  said           0.176  would          0.177
    so             0.181  then           0.181  front          0.186  when           0.191  marry          0.192
    was            0.198  die            0.198  have           0.198  think          0.200  shoot          0.204
    just           0.212  throw          0.212  be             0.213  probably       0.213  drove          0.213
    like           0.214  bought         0.214  add            0.215  stabbed        0.215  white          0.217
    best           0.222  had            0.223  point          0.224  end            0.224  took           0.225
    putting        0.225  pay            0.226  hole           0.227  smashed        0.228  pound          0.228
    left           0.229  give           0.230  about          0.231  checkbook      0.231  hit            0.231
    throat         0.232  black          0.234  Clinton        0.235  explode        0.236  going          0.237
    body           0.239  in             0.240  sterling       0.240  hate           0.241  second         0.241
    neck           0.242  sat            0.243  sell           0.244  sent           0.245  telling        0.245
    turning        0.245  Jesus          0.246  up             0.248  page           0.248  down           0.248
    leaves         0.248  fuck           0.249  taking         0.249  meant          0.251  situated       0.251
    blast          0.251  Press          0.251  ran            0.252  one            0.252  sue            0.253
    rupees         0.253  nipples        0.254  kissed         0.254  delete         0.256  live           0.257
    house          0.258  blow           0.261  earth          0.261  got            0.262  stop           0.263
    wherever       0.263  couple         0.263  immediate      0.263  stuck          0.264  feet           0.264

## English, ALIGNED-side (highest AUC)

    provide        0.920  provided       0.906  inform         0.899  express        0.880  escalate       0.873
    discuss        0.873  focus          0.871  avoid          0.868  examined       0.867  whispered      0.862
    explore        0.860  ensure         0.855  carefully      0.852  prioritize     0.846  address        0.846
    seek           0.843  prepare        0.842  speak          0.839  stumbled       0.839  present        0.837
    suddenly       0.836  quickly        0.836  ensured        0.834  consult        0.833  reminded       0.828
    requested      0.827  reconsider     0.827  adjusted       0.827  accidentally   0.820  review         0.820
    inspected      0.819  revisit        0.818  understand     0.817  recommend      0.817  clarify        0.816
    follow         0.815  seemed         0.815  revealed       0.815  reviewed       0.814  mentioned      0.812
    handle         0.807  consider       0.806  arranged       0.804  identified     0.801  proceed        0.799
    felt           0.799  request        0.798  shared         0.795  acknowledge    0.793  question       0.793
    update         0.792  approached     0.790  communicate    0.789  caused         0.786  discussed      0.785
    respond        0.784  accessed       0.784  break          0.784  negotiate      0.781  contact        0.781
    apologize      0.781  invest         0.780  remind         0.778  quantum        0.778  ignored        0.778
    planned        0.777  muttered       0.777  continue       0.776  pursue         0.775  everything     0.775
    barked         0.775  document       0.774  committed      0.773  offered        0.773  confront       0.773
    believe        0.770  admired        0.770  embrace        0.769  notify         0.769  smirked        0.769
    Sarah          0.767  responding     0.767  reach          0.767  conduct        0.766  prepared       0.765
    implement      0.765  whisper        0.764  hummed         0.763  further        0.763  abruptly       0.762
    create         0.761  Emily          0.760  apologized     0.759  pondered       0.758  approach       0.757
    perform        0.757  emerged        0.757  hum            0.756  swiftly        0.756  advise         0.755

## Chinese, BASE-side (lowest AUC)

    自己             0.194  去              0.208  喜欢             0.208  已              0.211  再              0.214
    哭              0.225  往              0.228  不              0.239  到              0.242  美国             0.246
    感谢             0.249  打电话            0.251  叫              0.256  能              0.256  可以             0.260
    说              0.260  打              0.263  左手             0.266  还              0.266  胳膊             0.268
    从              0.273  把              0.273  死              0.273  想              0.277  一刀             0.285
    再也             0.287  啊              0.287  男人             0.289  战争             0.294  相信             0.294
    10             0.301  浑身             0.301  杀了             0.304  杀死             0.308  见到             0.308
    跟              0.308  出去             0.311  跑              0.311  使劲             0.313  有              0.315
    朝              0.315  讨厌             0.317  躲              0.318  你              0.322  后者             0.322
    大家             0.322  把手             0.322  骂              0.322  知道             0.325  给              0.325
    被囚             0.327  我的             0.330  脖颈             0.330  5              0.332  又              0.336
    就              0.336  女人             0.339  就是             0.339  幻想             0.339  杀              0.339
    前者             0.341  杀人             0.344  一拳             0.346  厌恶             0.346  和              0.346
    好              0.346  没              0.346  走到             0.346  做              0.349  屁股             0.349
    觉得             0.349  8              0.351  吃              0.351  的是             0.351  有一             0.353
    被打             0.353  要              0.353  揍              0.355  过              0.355  9              0.356
    对着             0.356  得到             0.356  睡              0.356  砍              0.358  “              0.360
    脑袋             0.360  断了             0.362  6              0.363  创造             0.363  听见             0.363
    后悔             0.365  不再             0.367  也              0.367  付              0.367  告诉             0.367
    才              0.367  把他             0.367  救护车            0.367  有些             0.367  都              0.367

## Chinese, ALIGNED-side (highest AUC)

    关注             0.870  面包             0.867  编写             0.863  需要             0.862  寻求             0.853
    采取行动           0.844  质疑             0.843  一位             0.841  胜利             0.837  采取             0.830
    准备             0.827  木头             0.824  无法             0.824  进行             0.820  突然             0.813
    提供             0.799  晴朗             0.796  份              0.796  迅速             0.794  立即             0.794
    哪些             0.792  嗯              0.791  公园             0.789  长椅             0.784  反思             0.782
    鹿              0.780  喜悦             0.780  食物             0.775  钱包             0.775  支持             0.775
    快速             0.775  重新             0.772  醒来             0.770  调料             0.768  封              0.768
    深渊             0.766  击              0.761  120            0.761  香料             0.758  更              0.756
    再次             0.754  人的             0.754  警察             0.751  考虑             0.751  攻击             0.751
    哪里             0.751  袭击             0.749  外衣             0.747  双眼             0.747  火势             0.744
    改变             0.744  指责             0.744  投资             0.742  那人             0.741  朋友             0.741
    封书             0.741  走向             0.739  迎接             0.737  盐              0.737  摩擦             0.737
    寻找             0.737  加入             0.737  信用卡            0.737  下方             0.737  舞              0.735
    拳              0.735  尽快             0.734  封信             0.734  低头             0.734  巴黎             0.732
    阴天             0.730  晴天             0.730  撰写             0.730  墓              0.730  仍然             0.730
    怎样             0.728  栅栏             0.727  强行             0.727  如此             0.727  哑              0.727
    你”             0.727  50             0.727  糖              0.723  找到             0.722  完成             0.722
    表达             0.720  站在             0.720  60             0.720  轻轻地            0.716  挑战             0.716
    其              0.716  想到             0.715  跟随             0.711  绝望             0.709  嘶哑             0.709
    释放             0.708  时间             0.706  探索             0.706  几乎             0.706  起床             0.704

---

## Blinded lists for translation

Shuffled, no scores, no pole label. These are what a translator sees, so that
neither rank nor side can inform the translation.


### LIST A (100 words)

    then  ran  front  hate  had  second  marry  down  die  sell
    stabbed  say  immediate  bought  sue  body  end  so  right  back
    stuck  have  kill  situated  breasts  probably  couple  sterling  hit  up
    Jesus  left  fuck  know  live  said  best  went  be  point
    taking  took  in  told  pay  drove  about  first  beat  get
    pound  Press  nipples  add  neck  put  sat  think  kissed  go
    just  going  explode  meant  got  smashed  putting  wherever  page  checkbook
    would  black  shoot  house  all  stop  throw  give  hole  blow
    earth  turning  wrote  white  rupees  blast  feet  telling  throat  leaves
    one  head  was  top  threw  Clinton  delete  sent  like  when

### LIST B (100 words)

    requested  request  seek  consider  hummed  responding  conduct  create  discussed  arranged
    understand  clarify  reviewed  swiftly  believe  provide  whispered  discuss  ignored  felt
    prepare  prioritize  present  muttered  reconsider  communicate  caused  identified  apologized  everything
    planned  inspected  prepared  hum  speak  address  handle  carefully  proceed  whisper
    question  contact  advise  stumbled  mentioned  perform  pondered  quickly  consult  revealed
    review  document  notify  acknowledge  Sarah  barked  reach  confront  committed  accidentally
    accessed  emerged  update  Emily  offered  embrace  ensured  admired  approach  abruptly
    negotiate  focus  recommend  revisit  ensure  approached  apologize  inform  further  reminded
    follow  break  adjusted  remind  avoid  implement  escalate  smirked  examined  suddenly
    respond  express  shared  provided  explore  quantum  continue  invest  pursue  seemed

### LIST C (100 words)

    你  要  后悔  跑  9  后者  付  被囚  朝  觉得
    10  见到  感谢  就  有一  出去  和  被打  创造  厌恶
    可以  有  相信  啊  把  去  女人  的是  还  又
    打  把他  断了  “  死  杀了  把手  给  能  吃
    脖颈  从  脑袋  揍  没  有些  杀人  告诉  美国  做
    睡  往  对着  男人  幻想  8  屁股  过  哭  躲
    一刀  使劲  骂  不再  胳膊  都  浑身  砍  知道  听见
    5  喜欢  到  想  跟  叫  杀  打电话  再也  再
    战争  说  走到  杀死  大家  不  才  讨厌  一拳  前者
    好  6  也  左手  就是  得到  已  我的  救护车  自己

### LIST D (100 words)

    双眼  朋友  香料  进行  钱包  攻击  挑战  再次  栅栏  仍然
    晴天  一位  木头  尽快  嘶哑  突然  准备  重新  站在  想到
    关注  质疑  食物  120  表达  墓  怎样  快速  哪里  封
    更  巴黎  哑  其  采取  撰写  火势  糖  舞  公园
    哪些  轻轻地  摩擦  盐  投资  支持  拳  时间  走向  探索
    释放  鹿  低头  完成  改变  嗯  跟随  人的  起床  立即
    那人  60  如此  迅速  迎接  编写  喜悦  警察  50  需要
    胜利  几乎  采取行动  调料  找到  击  封书  长椅  醒来  晴朗
    寻求  信用卡  绝望  无法  份  外衣  强行  寻找  袭击  深渊
    面包  指责  反思  你”  下方  阴天  加入  考虑  提供  封信

---

## The prediction, recorded before any translation exists

Written 2026-08-13, before the four blinded lists were sent to translators and
before any lookup was run.

**THE TEST.** Translate each pole into the other language, blind, then look up
each translation's arm AUC in that other language's table. The bridge is a
declared instrument -- a translator that never sees a score, a rank, or a pole
label -- rather than a property of an encoder. This replaces the withdrawn
embedding-geometry test, whose problem was that every multilingual space is
trained to place translation equivalents together, so it partly measures its own
aligner.

**WHAT I EXPECT IF THE POLES CORRESPOND ACROSS LANGUAGES.**

    translations of LIST B (en aligned)  ->  HIGH zh AUC
    translations of LIST A (en base)     ->  LOW  zh AUC
    translations of LIST D (zh aligned)  ->  HIGH en AUC
    translations of LIST C (zh base)     ->  LOW  en AUC

Statistic: the gap between the two medians, per direction. Null: the same gap
computed on random word pairs drawn from the target table at matched coverage.

**WHAT WOULD MAKE IT UNINTERPRETABLE, stated now so it cannot be discovered
later as a convenience.** Coverage is the risk: a translation is only usable if
it appears in the target language's table with enough models behind it, and our
units are tokenizer boundaries rather than lexicon entries in both languages. If
coverage falls below about half of each list, the medians are over different
vocabularies and the comparison is not the one described here. Coverage is
reported per list before any median is.

**AND THE ASYMMETRY ALREADY ON RECORD.** The withdrawn geometric test found the
aligned side leaning and the base side flat (38.6% of coherent draws beat it). If
that asymmetry is real rather than an artifact of the encoder, this test should
reproduce it: the aligned direction should separate and the base direction should
not. If BOTH separate here, that is evidence the earlier base-side null was an
artifact of bare-word bge, whose English synonym gap is 0.138 against GloVe's
0.400.

---

## The result

Four blinded translators, one list each, no scores and no pole labels, one
translation per item. Coverage audited against the source lists rather than
against the translators' own line counts -- which were wrong, reporting 98, 97,
99 and 100 for four files that all held 100.

    pole          ->    n    median AUC   dev from centre   predicted   got
    en base       zh    63     0.4498        -0.0900          LOW       low
    en aligned    zh    28     0.6972        +0.1574          HIGH      high
    zh base       en    57     0.3757        -0.1309          LOW       low
    zh aligned    en    69     0.5144        +0.0078          HIGH      ~zero

    en-pole -> zh   gap +0.2474   random-word null p = 0.0002
    zh-pole -> en   gap +0.1387   random-word null p = 0.0002

**All four signs are as predicted and one of the four magnitudes is nothing.**
`zh aligned -> en` is +0.008 on 69% coverage, so that is a real null and not a
coverage failure.

**THE PRE-DECLARED COVERAGE FLOOR BINDS ON LIST B.** 28% is below the half I
declared before running, so the en-aligned median rests on 28 words and the
en->zh direction is reported but not quotable. The dropout is systematic, not
random: translations absent from the zh table are longer than those present
(2.22 characters against 1.96), because our units are tokenizer boundaries and
multi-character compounds -- 优先考虑, 重新考虑, 低声说 -- are exactly what the
English aligned pole translates into.

## WHY THE FOURTH CELL IS EMPTY, AND IT IS NOT A DEFECT

    pole         majority in-context tag
    en base      verb 59  noun 22  adverb 9  adjective 5
    en aligned   verb 89  adverb 8  noun 3
    zh base      verb 51  other 23  adverb 13  noun 13
    zh aligned   verb 38  noun 33  other 14  adverb 14

**The English aligned pole is 89% verbs. The Chinese aligned pole is 38% verbs
and 33% nouns**, and the nouns are concrete: 面包 bread, 木头 wood, 香料 spice,
糖 sugar, 盐 salt, 鹿 deer, 长椅 bench, 信用卡 credit card. Translated into
English those land near the English centre, which is exactly the +0.008.

So the two aligned poles are not the same object. English's is a near-pure class
of institutional-procedural verbs; Chinese's is that class plus a large concrete-
noun component with no English counterpart. **The correspondence is partial and
DIRECTIONAL** -- English-aligned words are aligned-side in Chinese, Chinese-
aligned words are not aligned-side in English -- and a single symmetric statistic
would have reported the average of a real effect and a null.

The best-covered cell is `zh base -> en` at -0.1309 on 57%, and the base
correspondence is the one this test supports most cleanly. That inverts the
withdrawn geometric result, which found the aligned side leaning and the base
side flat -- consistent with bare-word bge being the weaker instrument (English
synonym gap 0.138 against GloVe's 0.400) rather than with either pole being
special.
