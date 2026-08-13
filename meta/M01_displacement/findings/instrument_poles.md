# Instrument poles, one convention

Every instrument in P that scores a word, oriented the same way and drawn
from the SHARED vocabulary so the lists are comparable. Produced by
`k_instrument_poles.py`.

**POLE A is fall / base-side. POLE B is rise / aligned-side.** A faller is
pushed down by alignment, so it is high in base and low in aligned: the
movement and arm outcomes reach the same pole from opposite directions, and
the arm AUC is negated here to match. That is why P reports their agreement
as negative correlations.

    instrument           full vocab   outcome
    arm AUC (flipped)       3,866   ARM identity
    axis / GloVe            6,084   MOVEMENT
    axis / bge              6,120   MOVEMENT
    delta / bge-sub         4,064   MOVEMENT
    SHARED                  1,708   <- every list below

## Agreement

Spearman over the shared vocabulary, and overlap of the top-100 poles --
two instruments can correlate and still disagree about their extremes,
which is what a reader of a word list actually cares about.

    pair                                      rho     fall    rise
    arm AUC (flipped)  axis / GloVe       +0.465   30/100  23/100
    arm AUC (flipped)  axis / bge         +0.508   28/100  22/100
    arm AUC (flipped)  delta / bge-sub    +0.495   32/100  26/100
    axis / GloVe       axis / bge         +0.643   46/100  32/100
    axis / GloVe       delta / bge-sub    +0.562   30/100  23/100
    axis / bge         delta / bge-sub    +0.649   38/100  39/100

## arm AUC (flipped) -- POLE A -- fall / base-side

    went            told            kill            put             get             threw         
    go              say             know            wrote           said            marry         
    die             think           throw           drove           bought          add           
    took            putting         pay             smashed         give            hit           
    going           hate            sat             sell            sent            meant         
    situated        ran             sue             kissed          delete          live          
    blow            got             stop            stuck           came            stabbed       
    complain        erected         punched         tell            gave            lick          
    beat            drink           slashed         laid            fed             killed        
    jumped          cry             pulled          looked          come            see           
    ate             happened        asked           follows         left            melt          
    figured         suck            take            call            returning       pushed        
    read            dumped          make            occurred        turned          dropped       
    blown           raped           hope            slept           quit            fired         
    built           fuck            paid            resulted        guess           accumulated   
    waited          died            appealed        signed          smack           eating        
    dragged         fallen          opened          neared        

## arm AUC (flipped) -- POLE B -- rise / aligned-side

    provide         provided        inform          express         escalate        discuss       
    avoid           examined        whispered       explore         ensure          prioritize    
    address         seek            prepare         speak           stumbled        ensured       
    consult         reminded        requested       reconsider      adjusted        inspected     
    revisit         understand      recommend       clarify         follow          seemed        
    revealed        reviewed        mentioned       handle          consider        arranged      
    identified      proceed         felt            acknowledge     approached      communicate   
    caused          discussed       respond         accessed        negotiate       apologize     
    invest          remind          ignored         planned         muttered        continue      
    pursue          barked          committed       offered         confront        believe       
    admired         embrace         notify          smirked         responding      reach         
    conduct         prepared        implement       hummed          create          apologized    
    pondered        perform         emerged         advise          presented       hesitated     
    engage          failed          claimed         manipulate      managed         suggest       
    secured         checked         scattered       propose         verified        vanished      
    faced           scanned         investigate     need            submitted       observed      
    exchanged       spoke           browsed         exhaled       

## axis / GloVe -- POLE A -- fall / base-side

    drop            fuck            whacked         gets            dump            blow          
    drink           suck            dumped          lick            blowing         bought        
    ate             stabbed         goes            peel            bury            stabbing      
    knock           shoot           dropping        stuck           smacked         throw         
    smack           got             drank           slammed         eat             marry         
    pulls           hate            drag            slamming        knocked         blown         
    died            pick            dragging        sucked          shove           quote         
    nailed          slapped         ripped          knocking        deduct          bore          
    pulling         grated          screaming       rub             quit            go            
    sue             stuffed         stick           rained          hooked          rip           
    landed          spanked         kissing         blew            guess           kill          
    pull            purchased       stay            yelling         roll            dropped       
    haul            pulled          miss            driving         kicked          going         
    apologise       hitting         ran             dragged         saw             went          
    punching        murdered        beaten          flipped         thrashing       screwed       
    grind           sold            hurry           peeled          gave            tell          
    fucked          rubbing         count           lose          

## axis / GloVe -- POLE B -- rise / aligned-side

    explore         automate        activated       navigated       addressing      evaluate      
    communicate     pondered        paused          create          seemed          confronting   
    acknowledging   prioritize      respond         unionize        forged          sensed        
    discussed       disturb         highlighted     manipulate      forgotten       verified      
    confront        develop         conduct         engage          reflect         emphasize     
    capture         reviewed        transformed     guarded         escalate        organize      
    frowned         marveled        betray          discuss         distributed     chatted       
    emerged         applauded       discussing      demonstrated    promote         describe      
    embraced        demonstrate     confronted      establish       combed          winced        
    transform       unfolded        verify          contribute      examine         involve       
    observed        calm            startled        resolve         acknowledged    focused       
    saluted         reassured       retreated       grasped         address         poised        
    conducted       sharpen         falsified       scanned         awaited         tested        
    assessed        hidden          ignored         fade            created         invoked       
    chuckled        giggled         snuggled        participate     protected       provide       
    possess         viewed          shuffled        blinked         monitored       avoided       
    responding      hyperventilate  unlocked        praised       

## axis / bge -- POLE A -- fall / base-side

    fuck            fucked          stabbing        killed          hitting         blow          
    suck            kick            murdered        knocking        blowing         punching      
    kill            rubbing         dump            died            licking         pulling       
    dumping         stabbed         bake            blown           die             eat           
    kicked          smack           pulls           pulled          kissing         sucked        
    pull            bought          smacked         throwing        hit             throw         
    licked          eating          goes            slaughtered     puke            knocked       
    feed            raped           paid            quote           toss            drink         
    roll            tumble          punched         loaded          putting         burn          
    knock           rammed          dug             put             stomp           whacked       
    beaten          slapped         beg             laid            dropping        download      
    hammered        deduct          swings          went            means           chucked       
    hurl            pumped          won             rolled          falling         drank         
    downloaded      come            bury            dig             bear            beat          
    grind           hung            weigh           sold            stealing        rocked        
    climb           buy             tossing         quit            blew            fired         
    pay             cooked          pummel          gotten        

## axis / bge -- POLE B -- rise / aligned-side

    communicate     navigated       explore         shielded        confronted      contemplated  
    confronting     pondered        confront        create          reflect         encountered   
    grimaced        created         echoed          perform         challenged      hesitated     
    monitored       realize         represented     crossed         expressed       clarify       
    approached      unfolded        rearranged      sifted          realizing       formed        
    responding      understood      prepared        signaled        shivered        noticed       
    sharpen         reflected       guarded         noticing        protect         sneered       
    meditate        embrace         prioritize      intoned         implement       identified    
    gestured        emerged         neared          demonstrated    whispered       develop       
    frowned         understand      meditated       uncovered       recognized      shone         
    disappearing    performed       stirred         examine         maintained      existed       
    demonstrate     straddled       prepare         embraced        hidden          administer    
    disappeared     established     squinted        addressing      engage          presented     
    reconnected     secured         evaluate        acknowledging   arrange         unfurled      
    unearthed       discovered      contradicted    considered      concentrate     rethink       
    forgotten       steered         connect         appeared        cradled         activated     
    fumbled         assessed        forged          disappear     

## delta / bge-sub -- POLE A -- fall / base-side

    reared          punching        stabbing        kick            throwing        shaved        
    fucked          rode            stabbed         blow            jack            rubbing       
    go              shows           happened        getting         gotten          throw         
    threw           punched         suck            put             hitting         means         
    totaled         killed          went            retract         smacked         slip          
    downed          refuted         drop            goes            got             cast          
    smack           rubbed          says            objected        come            die           
    win             shave           rub             guess           scrape          ate           
    planted         contradicted    jabbed          patted          kill            ejaculated    
    click           fuck            sucked          kicked          pushing         typed         
    sit             hit             clicked         murdered        pulls           won           
    falling         eating          undid           beat            fry             meant         
    stands          told            get             pay             rip             blown         
    injected        stroking        shoved          pushed          whacked         run           
    came            wrecked         includes        sent            flung           addresses     
    lifting         slashed         landed          attacking       drove           jump          
    dropping        rocked          topple          buried        

## delta / bge-sub -- POLE B -- rise / aligned-side

    qualify         contemplated    engage          evaluate        murmured        meditated     
    pondered        express         consider        implement       whispered       assessed      
    reflected       translated      meditate        mumbled         pursue          translate     
    unleash         verify          snarled         whirring        establish       expressed     
    possess         activated       possessed       connect         sharpen         unfurled      
    fumbled         ensure          communicate     snorted         surveyed        involve       
    discuss         revise          squinted        honked          reflect         explore       
    taunt           negotiate       growled         need            smoothed        reconsider    
    grimaced        neared          trapped         grunted         address         winked        
    frowned         approaching     skimmed         inspected       hummed          notify        
    echoed          necessitated    documented      unzip           constituted     hesitated     
    ensured         prostrated      overheard       addressing      monitored       situated      
    needed          sneered         taunted         clarify         acknowledge     lunged        
    examined        maintained      encountered     recognized      verified        sifted        
    proceeding      enjoyed         pausing         winced          disarmed        considered    
    determine       realizing       examine         appreciate      noticing        erupted       
    approached      unfolded        conduct         intervene     
