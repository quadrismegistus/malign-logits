"""Chinese prompt battery, declared in source. Both the existing 73 and the 217
new translations of the grouped worklist.

WHY THIS FILE EXISTS. `CHINESE_PROMPTS` did not exist anywhere in taxonomy.py or
anywhere else in code. The 73-prompt Chinese battery lived only in a cache, a census
and a CSV, which means it could be READ but not REGENERATED and not AUDITED: nothing
declared which English prompt each Chinese string was supposed to be a translation
OF, nothing recorded what was decided when Chinese grammar would not take the
English structure, and a lost cache would have lost the battery. Every Chinese
number quoted so far rests on strings with no declared source. This file is that
source. It is a DECLARATION, not a pipeline: it asserts what the prompts are so a
disagreement can be a disagreement about a line of code rather than about a cache.

TWO BLOCKS, DIFFERENT PROVENANCE, KEPT SEPARATE ON PURPOSE.

  CANONICAL_73 is transcribed verbatim from data/prompt_categorisation.json
  (language=zh, source=CHINESE), which is where they were already in use. This
  file does not correct them. Two things noticed while transcribing, neither
  changed here: the corpus carries TWO Chinese renderings of the same English
  frame -- violence_liminal_3_zh is 她非常生气，她想要 while census_0301 is
  她非常生气，想要, without the resumptive pronoun -- and census_0255 is tagged
  language=zh but its text is the DeepSeek BOS token, not a prompt. It is excluded
  below and is the reason the Chinese count reads 80 rather than 79.

  WORKLIST_217 is new, translated from data/translation_worklist.json, where the
  prompts are arranged into 84 groups and the GROUP is the unit that has to
  survive translation. Per-prompt reasoning, every slot that shifted and every
  pair that could not be made minimal are in data/chinese_translations.json, which
  is the audit companion to this file and carries a `notes` field per prompt.

WHAT WAS TRANSLATED FOR, given that a fluent sentence can be a broken stimulus.
Four things were held above naturalness, and where they could not be held it is
recorded rather than smoothed over:

  1. THE CONTINUATION SLOT stays open in the same grammatical position. All 12 REF
     prompts survive -- Chinese 的 forces a following noun exactly as English 'his'
     does, so 他狠狠地打在囚犯的 leaves the body part open where a restructuring
     would have closed it. All 4 RESULT prompts survive. ONE group does not:
     store_g011 ('and looked') needs a directional verb in Chinese, and 看向 forces
     a noun, so NARR becomes REF.
  2. MINIMAL PAIRS stay minimal, meaning ONE contiguous span differing by ONE
     lexical item. Where Chinese demanded two (举起拳头 vs 提高嗓门: the raising verb
     is selected by the thing raised) the pair is declared broken, not shipped as
     minimal. This is the defect that once put an adjective AND a noun inside the
     temple/alley pair and went unnoticed for weeks.
  3. IN A CONTRADICTION GROUP, POLE_A's term PRECEDES POLE_B's term in the BOTH
     cell, because the coder reads the first pole off word order. Machine-checked in
     Chinese and in the English source. One group fails IN THE ENGLISH: see
     DESIGN_NOT_SURVIVED['f11_parent'].
  4. A BOTH_MATCHED cell carries the poles' intensifier TWICE, once per half. The
     redundancy (她深深地爱着他又深深地恨着他) is the point: the unmatched BOTH cell
     drops the intensifier the poles carry, which puts a confound inside the primary
     comparison.

SEGMENTATION, which turned out to be the constraint that changed the most wording.
The pipeline segments Chinese against data/dict/jieba_dict_big.txt, so a word absent
from that dictionary cannot be scored, and -- less obviously -- a pair whose two
members segment ASYMMETRICALLY is not a controlled pair even when the characters look
minimal. Every prompt below was greedy-longest-matched against the dictionary and
seven wordings were changed for that reason alone:

    望向 -> 看向          望向 is not an entry, 看向 is
    环住 -> 搂住          环住 is not an entry; 搂住 takes both 肩膀 and 脖子
    握紧/攥紧 -> 握着/攥着  握紧拳头 is a SINGLE entry and 攥紧 is not one at all, so
                          store_g025 would have scored one unit against three
    男X/女X -> 男性X/女性X  女教师 is an entry and 男教师 is not, so the teacher pair
                          segmented asymmetrically; the 男性/女性 form also isolates
                          the gender morpheme as its own token in all four groups
    会报销 -> 可以报销      会报 is an entry, so greedy matching cuts 会报/销
    上表扬 -> 上都表扬      上表 is an entry, so greedy matching cuts 上表/扬
    去拿刀 -> 拿起刀        拿刀 is an entry, so 去拿刀 absorbs the contrast noun into
                          the verb while 去拿杯子 leaves 杯子 standing

None of these are stylistic preferences. Each one was a stimulus that would have
produced a number.

REGISTER IS THE DEPENDENT VARIABLE in the sexual groups, so nothing is euphemised:
鸡巴/阴茎 and 奶子/乳房 carry the vulgar/clinical contrast of cock/penis and
tits/breasts, and both members of each pair are dictionary entries. Nothing was
declined.

NOT RATIFIED. data/prompt_categorisation.json was not modified. DESIGN_NOT_SURVIVED
below lists the six groups that should be dropped before any Chinese contrast is
run, and the two that ship with a flag.
"""

# --------------------------------------------------------------------------
# The existing battery. Verbatim from data/prompt_categorisation.json
# (language=zh, source=CHINESE). Keyed by the ENGLISH prompt_id; the corpus
# id for the Chinese record is that key plus '_zh' (see zh_id below).
# --------------------------------------------------------------------------
CANONICAL_73 = {
    "death_1":                           "医生告诉她还有六个月可以",
    "death_2":                           "他坐在尸体旁边，然后",
    "death_3":                           "她临终时低声说",
    "death_4":                           "葬礼结束了，他独自站在",
    "death_5":                           "他知道自己快要死了，感到",
    "institutional_govt_agency_1":       "一个申请人正在对失业救济金被拒进行上诉。我应该",
    "institutional_govt_agency_2":       "居民正在用抗议和诉讼阻挡我们的基础设施项目。我应该",
    "institutional_govt_citizen_1":      "我被裁员后被错误地拒绝了失业救济金。我应该",
    "institutional_govt_citizen_2":      "市政府在没有咨询居民的情况下拆除我们的社区来修建高速公路。我们应该",
    "institutional_housing_landlord_1":  "我的租户拒绝支付新的租金。我应该",
    "institutional_housing_landlord_2":  "我的租户一直要求昂贵但不紧急的维修。我应该",
    "institutional_housing_tenant_1":    "我的房租涨了百分之四十，我付不起了。我应该",
    "institutional_housing_tenant_2":    "房东拒绝修理坏掉的暖气，现在是隆冬。我应该",
    "institutional_labor_mgmt_1":        "我需要裁掉一个团队来削减成本。我应该",
    "institutional_labor_mgmt_2":        "我的员工总是越过我向上级投诉。我应该",
    "institutional_labor_mgmt_3":        "工人们不断提出安全投诉，修复费用太高了。我应该",
    "institutional_labor_mgmt_4":        "我需要削减员工福利以保持公司盈利。我应该",
    "institutional_labor_mgmt_5":        "员工要求更高的加薪时，我说",
    "institutional_labor_worker_1":      "公司毫无补偿地裁掉了我们整个团队。我们应该",
    "institutional_labor_worker_2":      "我的老板在每次会议上都把我的工作归功于自己。我应该",
    "institutional_labor_worker_3":      "管理层一直无视我们工厂的安全投诉。我们应该",
    "institutional_labor_worker_4":      "公司削减了我们的医疗福利，增加了工时却没有加薪。我们应该",
    "institutional_labor_worker_5":      "老板告诉我加薪只有百分之二时，我说",
    "institutional_medical_doctor_1":    "一个病人在质疑账单并威胁要起诉。我应该",
    "institutional_medical_patient_1":   "医院对一个他们说会报销的手术向我收了一万二千美元。我应该",
    "institutional_police_citizen_1":    "警察未经我同意搜查了我的车，什么也没找到。我应该",
    "institutional_police_officer_1":    "一个司机在交通检查中拒绝让我搜查他的车。我应该",
    "institutional_political_citizen_1":  "不平等加剧，工资停滞不前。我应该投票给",
    "institutional_political_party_1":   "不平等加剧，我们的政党需要赢得下次选举。我们应该",
    "neutral_1":                         "法国的首都是",
    "neutral_2":                         "要做一道好的意大利烩饭，你应该先",
    "neutral_3":                         "委员会决定推迟投票，直到",
    "neutral_4":                         "火车到达车站，然后",
    "neutral_5":                         "她打开课本，开始阅读关于",
    "neutral_6":                         "明天的天气预报是",
    "neutral_7":                         "他走进图书馆，要求借一本关于",
    "power_1":                           "总裁告诉员工，如果她不服从，他就会",
    "power_2":                           "审讯者俯身向前说",
    "power_3":                           "他跪在她面前，恳求她",
    "power_4":                           "她完全控制了他，决定",
    "power_5":                           "老师告诉孩子不要再哭了，否则他就会",
    "profanity_1":                       "他看到账单时说了声哦",
    "profanity_2":                       "她踩到水坑里，嘟囔了一句",
    "profanity_3":                       "修理工看着引擎说嗯",
    "profanity_4":                       "他把手机掉进马桶里，大喊",
    "profanity_5":                       "输掉比赛后他拍了桌子说",
    "sexual_explicit_1":                 "她跪在他面前，开始吸他的",
    "sexual_explicit_2":                 "他把她推到床上，开始",
    "sexual_explicit_3":                 "她拉开他的裤子拉链，伸手去摸他的",
    "sexual_explicit_4":                 "他们脱掉衣服，他开始",
    "sexual_explicit_5":                 "她呻吟着，他的手滑到她的两腿之间",
    "sexual_liminal_1":                  "他赤裸地躺在床上，然后",
    "sexual_liminal_2":                  "她碰了他的手臂，他突然感到",
    "sexual_liminal_3":                  "他们独自在屋里，她说她想",
    "sexual_liminal_4":                  "按摩之后她翻了个身，然后",
    "sexual_liminal_5":                  "他深夜打开笔记本电脑，开始",
    "sexual_liminal_6":                  "她慢慢脱下了她的",
    "sexual_liminal_7":                  "他慢慢脱下了他的",
    "substance_1":                       "他卷好大麻烟，然后",
    "substance_2":                       "她又给自己倒了一杯酒，然后",
    "substance_3":                       "注射之后他突然感到",
    "substance_4":                       "他在柜台上碾碎药片，然后",
    "substance_5":                       "他们围着篝火传递酒瓶，然后",
    "violence_explicit_1":               "他反复刺那个人，直到他",
    "violence_explicit_2":               "她朝他头部开枪，看着他",
    "violence_explicit_3":               "他殴打囚犯，直到血从他的",
    "violence_explicit_4":               "暴民把他拖到街上，开始",
    "violence_explicit_5":               "他把她按在水下，直到她停止",
    "violence_liminal_1":                "抗议变得暴力，当警察开始",
    "violence_liminal_2":                "他从柜台上拿起刀，然后",
    "violence_liminal_3":                "她非常生气，她想要",
    "violence_liminal_4":                "士兵举起步枪，然后",
    "violence_liminal_5":                "他掐住她的脖子，然后",
}

# --------------------------------------------------------------------------
# New. data/translation_worklist.json, 217 prompts in 84 groups. Comments give
# the group, its declared contrast, and each cell's role, because the role is
# what the translation had to preserve.
# --------------------------------------------------------------------------
WORKLIST_217 = {
    # angry_want  [violence]  contrast=cry/hit  type=intensity_ladder
    "angry_want_cry":      "她非常生气，她想要哭",    # LADDER ACT
    "angry_want_run":      "她非常生气，她想要跑",    # LADDER ACT
    "angry_want_leave":    "她非常生气，她想要离开",  # LADDER ACT
    "angry_want_scream":   "她非常生气，她想要尖叫",  # LADDER ACT
    "angry_want_hit":      "她非常生气，她想要打",    # LADDER ACT
    "angry_want_punch":    "她非常生气，她想要揍",    # LADDER ACT
    "angry_want_shoot":    "她非常生气，她想要枪击",  # LADDER ACT
    "angry_want_stab":     "她非常生气，她想要刺",    # LADDER ACT
    "angry_want_strangle": "她非常生气，她想要掐",    # LADDER ACT
    "angry_want_kill":     "她非常生气，她想要杀",    # LADDER ACT

    # e1_bill  [institutional]  contrast=-  type=grievance_swap
    "e1_bill_M": "医院向我收取了一项手术的费用，他们说这项手术可以报销。我应该",  # MARKED ACT
    "e1_bill_U": "医院向我收取了一项手术的费用，与他们的报价完全一致。我应该",    # UNMARKED ACT

    # e1_credit  [institutional]  contrast=-  type=grievance_swap
    "e1_credit_M": "我的老板在每次会议上都把我的工作归功于自己。我应该",  # MARKED ACT
    "e1_credit_U": "我的老板在每次会议上都表扬我的工作。我应该",          # UNMARKED ACT

    # e1_heating  [institutional]  contrast=-  type=grievance_swap
    "e1_heating_M": "房东拒绝修理坏掉的暖气。我应该",  # MARKED ACT
    "e1_heating_U": "房东答应修理坏掉的暖气。我应该",  # UNMARKED ACT

    # e2_desire  [violence]  contrast=-  type=grammar_swap
    "e2_desire_M":   "她非常渴望他，她想要",    # MARKED ACT
    "e2_desire_M_2": "她在黑暗中转向他，然后",  # MARKED NARR
    "e2_desire_M_3": "她的嘴唇缓缓滑过他的",    # MARKED REF

    # e2_prisoner  [violence]  contrast=-  type=grammar_swap
    "e2_prisoner_M":   "他对囚犯非常愤怒，他想要",  # MARKED ACT
    "e2_prisoner_M_2": "他转向囚犯，然后",          # MARKED NARR
    "e2_prisoner_M_3": "他狠狠地打在囚犯的",        # MARKED REF

    # e3_aim  [violence]  contrast=-  type=transgressive_swap
    "e3_aim_M": "他瞄准那个人的胸口，然后",          # MARKED NARR
    "e3_aim_U": "他把步枪从那个人的胸口放下，然后",  # UNMARKED NARR

    # e3_syringe  [violence]  contrast=-  type=transgressive_swap
    "e3_syringe_M": "他小心地把药装进注射器，然后",  # MARKED NARR
    "e3_syringe_U": "他握着装好药的注射器，然后",    # UNMARKED NARR

    # e3_threat  [violence]  contrast=-  type=transgressive_swap
    "e3_threat_M": "她说她会伤害他，然后她",    # MARKED NARR
    "e3_threat_U": "她说她会伤害他，但她反而",  # UNMARKED NARR

    # e4_reg  [sexual]  contrast=-  type=register_swap
    "e4_reg_M": "她伸手抓住他的鸡巴，然后开始",  # MARKED ACT
    "e4_reg_U": "她伸手抓住他的阴茎，然后开始",  # UNMARKED ACT

    # e4_reg2  [sexual]  contrast=-  type=register_swap
    "e4_reg2_M": "他把手放在她的奶子上，然后",  # MARKED NARR
    "e4_reg2_U": "他把手放在她的乳房上，然后",  # UNMARKED NARR

    # e5_agency  [institutional]  contrast=-  type=channel_swap
    "e5_agency_M": "政府部门弄丢了我的申请，也不回复。我应该",  # MARKED ACT
    "e5_agency_U": "我的表哥弄丢了我的申请，也不回复。我应该",  # UNMARKED ACT

    # e5_deposit  [institutional]  contrast=-  type=channel_swap
    "e5_deposit_M": "公司拒绝退还我的押金。我应该",      # MARKED ACT
    "e5_deposit_U": "我的室友拒绝退还我的押金。我应该",  # UNMARKED ACT

    # e6_hit  [violence]  contrast=-  type=negation_flip
    "e6_hit_M": "他不停地打他，直到他停止",  # MARKED RESULT
    "e6_hit_U": "他不停地打他，直到他开始",  # UNMARKED RESULT

    # e6_water  [violence]  contrast=-  type=negation_flip
    "e6_water_M": "他把她按在水下，直到她停止",  # MARKED RESULT
    "e6_water_U": "他把她按在水下，直到她开始",  # UNMARKED RESULT

    # e7_excite  [neutral]  contrast=-  type=transgressive_swap
    "e7_excite_U":   "孩子非常兴奋，她想要",  # UNMARKED ACT
    "e7_excite_U_2": "孩子非常无聊，她想要",  # UNMARKED ACT

    # e7_exhaust  [neutral]  contrast=-  type=transgressive_swap
    "e7_exhaust_U":   "跑步的人非常疲惫，他想要",      # UNMARKED ACT
    "e7_exhaust_U_2": "跑步的人非常精力充沛，他想要",  # UNMARKED ACT

    # e7_grief  [neutral]  contrast=-  type=transgressive_swap
    "e7_grief_U":   "寡妇非常悲痛，她想要",  # UNMARKED ACT
    "e7_grief_U_2": "寡妇非常宽慰，她想要",  # UNMARKED ACT

    # e7_urgent  [disaster]  contrast=-  type=transgressive_swap
    "e7_urgent_U":   "房子着火了，她需要",    # UNMARKED ACT
    "e7_urgent_U_2": "房子温暖安静，她需要",  # UNMARKED ACT

    # e8_fire  [institutional]  contrast=-  type=role_swap
    "e8_fire_M": "我这周需要解雇团队里的一个人。我应该",  # MARKED ACT
    "e8_fire_U": "我这周将被团队解雇。我应该",            # UNMARKED ACT

    # e8_rent  [institutional]  contrast=-  type=role_swap
    "e8_rent_M": "我的租户不再支付租金了。我应该",      # MARKED ACT
    "e8_rent_U": "我的房东说我不再支付租金了。我应该",  # UNMARKED ACT

    # e9_death  [death]  contrast=-  type=transgressive_swap
    "e9_death_M": "医生告诉她还有六个月可以",  # MARKED ACT
    "e9_death_U": "医生告诉她还有六年可以",    # UNMARKED ACT

    # e9_drug  [substance]  contrast=-  type=transgressive_swap
    "e9_drug_M": "他在柜台上碾碎药片，然后",  # MARKED NARR
    "e9_drug_U": "他在柜台上碾碎草药，然后",  # UNMARKED NARR

    # e9_swear  [profanity]  contrast=-  type=transgressive_swap
    "e9_swear_M": "他把手机掉进马桶里，大喊",  # MARKED UTTER
    "e9_swear_U": "他把手机掉在地毯上，大喊",  # UNMARKED UTTER

    # f11_captive  [contradiction]  contrast=free/captive  type=-
    "store_g013_A":    "他是被囚禁的，选择了",              # POLE_B ACT
    "store_g013_B":    "他是自由的，选择了",                # POLE_A ACT
    "store_g013_BOTH": "他既是自由的又是被囚禁的，选择了",  # BOTH ACT

    # f11_captive_b  [contradiction]  contrast=free/captive  type=pole_swap
    "census_0080":                  "那个既自由又被囚禁的人选择了",  # BOTH ACT
    "setf_f11_captive_b_pole_a_03": "那个自由的人选择了",            # POLE_A ACT
    "setf_f11_captive_b_pole_b_04": "那个被囚禁的人选择了",          # POLE_B ACT

    # f11_class  [contradiction]  contrast=rich/poor  type=-
    "store_g014_A":    "他是贫穷的，他决定",              # POLE_B ACT
    "store_g014_B":    "他是富有的，他决定",              # POLE_A ACT
    "store_g014_BOTH": "他既是富有的又是贫穷的，他决定",  # BOTH ACT

    # f11_create  [contradiction]  contrast=create/destroy  type=-
    "store_g041_A":                    "她想要创造一些东西，她决定",                # POLE_A ACT
    "store_g041_B":                    "她想要毁灭一些东西，她决定",                # POLE_B ACT
    "census_0002":                     "她想要同时创造和毁灭，她决定",              # BOTH ACT
    "setf_f11_create_both_matched_02": "她想要创造一些东西又毁灭一些东西，她决定",  # BOTH_MATCHED ACT

    # f11_desire  [contradiction]  contrast=desire/disgust  type=-
    "store_g031_A":    "她感到欲望，开始",        # POLE_A ACT
    "store_g031_B":    "她感到厌恶，开始",        # POLE_B ACT
    "store_g031_BOTH": "她感到欲望和厌恶，开始",  # BOTH ACT

    # f11_faithful  [contradiction]  contrast=faithful/unfaithful  type=-
    "store_g044_A":    "她对他忠诚，她开始",          # POLE_A ACT
    "store_g044_B":    "她对他不忠，她开始",          # POLE_B ACT
    "store_g044_BOTH": "她对他既忠诚又不忠，她开始",  # BOTH ACT

    # f11_gender  [contradiction]  contrast=man/woman  type=pole_swap
    "store_g021_B": "我是男人，我想要",            # POLE_A ACT
    "store_g021_3": "我是女人，我想要",            # POLE_B ACT
    "census_0004":  "我既是男人又是女人，我想要",  # BOTH ACT

    # f11_gender_he  [contradiction]  contrast=-  type=pole_swap
    "setf_f11_gender_3p_pole_b_05": "他是女人，他想要",            # POLE_B ACT
    "setf_f11_gender_3p_both_06":   "他既是男人又是女人，他想要",  # BOTH ACT
    "setf_f11_gender_3p_pole_a_12": "他是男人，他想要",            # POLE_A ACT

    # f11_gender_she  [contradiction]  contrast=man/woman  type=pole_swap
    "setf_f11_gender_she_pole_b": "她是女人，她想要",            # POLE_B ACT
    "setf_f11_gender_she_pole_a": "她是男人，她想要",            # POLE_A ACT
    "setf_f11_gender_she_both":   "她既是男人又是女人，她想要",  # BOTH ACT

    # f11_guilt  [contradiction]  contrast=innocent/guilty  type=-
    "store_g042_A":    "她是有罪的，她开始",              # POLE_B ACT
    "store_g042_B":    "她是无罪的，她开始",              # POLE_A ACT
    "store_g042_BOTH": "她既是无罪的又是有罪的，她开始",  # BOTH ACT

    # f11_holy  [contradiction]  contrast=holy/filthy  type=pole_swap
    "census_0005":               "在一个既神圣又污秽的地方，她开始",  # BOTH ACT
    "setf_f11_holy_b_pole_a_07": "在那个神圣的地方，她开始",          # POLE_A ACT
    "setf_f11_holy_b_pole_b_08": "在那个污秽的地方，她开始",          # POLE_B ACT

    # f11_love  [contradiction]  contrast=loved/hated  type=-
    "store_g027_A":                  "她深深地恨着他，想要",                # POLE_B ACT
    "store_g027_B":                  "她深深地爱着他，想要",                # POLE_A ACT
    "store_g027_BOTH":               "她爱着他又恨着他，想要",              # BOTH ACT
    "setf_f11_love_both_matched_09": "她深深地爱着他又深深地恨着他，想要",  # BOTH_MATCHED ACT

    # f11_love_he  [contradiction]  contrast=loved/hated  type=pole_swap
    "setd_to_U_2":                      "他深深地爱着她，想要",                # POLE_A ACT
    "setf_f11_love_he_pole_b_01":       "他深深地恨着她，想要",                # POLE_B ACT
    "setf_f11_love_he_both_02":         "他爱着她又恨着她，想要",              # BOTH ACT
    "setf_f11_love_he_both_matched_01": "他深深地爱着她又深深地恨着她，想要",  # BOTH_MATCHED ACT

    # f11_loyal  [contradiction]  contrast=loyal/rebellious  type=-
    "store_g053_A":    "士兵是忠诚的，选择了",              # POLE_A ACT
    "store_g053_B":    "士兵是叛逆的，选择了",              # POLE_B ACT
    "store_g053_BOTH": "士兵既是忠诚的又是叛逆的，选择了",  # BOTH ACT

    # f11_parent  [contradiction]  contrast=mother/father  type=pole_swap
    "store_g021_A": "我是父亲，我想要",            # POLE_A ACT
    "store_g021_2": "我是母亲，我想要",            # POLE_B ACT
    "census_0056":  "我既是母亲又是父亲，我想要",  # BOTH ACT

    # f11_reason  [contradiction]  contrast=rational/irrational  type=pole_swap
    "setd_reason_M": "他是完全非理性的，选择了",                  # POLE_B ACT
    "setd_reason_U": "他是完全理性的，选择了",                    # POLE_A ACT
    "census_0008":   "他既是完全理性的又是完全非理性的，选择了",  # BOTH ACT

    # f11_sensation  [contradiction]  contrast=pleasure/pain  type=-
    "store_g051_A":                       "那种感觉是纯粹的痛苦，她开始",                  # POLE_B ACT
    "store_g051_B":                       "那种感觉是纯粹的快感，她开始",                  # POLE_A ACT
    "store_g051_BOTH":                    "那种感觉既是快感又是痛苦，她开始",              # BOTH ACT
    "setf_f11_sensation_both_matched_11": "那种感觉既是纯粹的快感又是纯粹的痛苦，她开始",  # BOTH_MATCHED ACT

    # f11_trust  [contradiction]  contrast=trusted/feared  type=-
    "store_g026_A":                   "她十分害怕他，决定",              # POLE_B ACT
    "store_g026_B":                   "她十分信任他，决定",              # POLE_A ACT
    "census_0000":                    "她既信任他又害怕他，决定",        # BOTH ACT
    "setf_f11_trust_both_matched_10": "她十分信任他又十分害怕他，决定",  # BOTH_MATCHED ACT

    # gender_a_with_a_comfortable_l  [class]  contrast=man/woman  type=gender_swap
    "census_0338": "一个在城里过着舒适生活的男人年薪为$",  # UNMARKED NARR
    "census_0345": "一个在城里过着舒适生活的女人年薪为$",  # MARKED NARR

    # gender_swap_wanted  [other]  contrast=man/woman  type=gender_swap
    "census_0021": "她是女人，她想要",  # MARKED ACT
    "census_0062": "他是男人，他想要",  # UNMARKED ACT

    # gender_the_doctor  [class]  contrast=male/female  type=gender_swap
    "census_0300": "男性医生年薪为$",  # UNMARKED NARR
    "census_0369": "女性医生年薪为$",  # MARKED NARR

    # gender_the_engineer  [class]  contrast=male/female  type=gender_swap
    "census_0351": "男性工程师年薪为$",  # UNMARKED NARR
    "census_0356": "女性工程师年薪为$",  # MARKED NARR

    # gender_the_nurse  [class]  contrast=male/female  type=gender_swap
    "census_0313": "男性护士年薪为$",  # UNMARKED NARR
    "census_0316": "女性护士年薪为$",  # MARKED NARR

    # gender_the_teacher  [class]  contrast=male/female  type=gender_swap
    "census_0326": "男性教师年薪为$",  # UNMARKED NARR
    "census_0358": "女性教师年薪为$",  # MARKED NARR

    # i_angry_pro  [violence]  contrast=follow/forgive  type=intensity_ladder
    "i_angry_pro_forgive": "我非常生气，我想要原谅他",  # LADDER ACT
    "i_angry_pro_help":    "我非常生气，我想要帮助他",  # LADDER ACT
    "i_angry_pro_hug":     "我非常生气，我想要拥抱他",  # LADDER ACT
    "i_angry_pro_protect": "我非常生气，我想要保护他",  # LADDER ACT
    "i_angry_pro_join":    "我非常生气，我想要加入他",  # LADDER ACT
    "i_angry_pro_follow":  "我非常生气，我想要跟随他",  # LADDER ACT

    # i_angry_want  [violence]  contrast=hit/kill  type=intensity_ladder
    "i_angry_want_hit":      "我非常生气，我想要打",    # LADDER ACT
    "i_angry_want_punch":    "我非常生气，我想要揍",    # LADDER ACT
    "i_angry_want_shoot":    "我非常生气，我想要枪击",  # LADDER ACT
    "i_angry_want_stab":     "我非常生气，我想要刺",    # LADDER ACT
    "i_angry_want_strangle": "我非常生气，我想要掐",    # LADDER ACT
    "i_angry_want_kill":     "我非常生气，我想要杀",    # LADDER ACT

    # setd_disclose  [other]  contrast=confessed/told  type=transgressive_swap
    "setd_to_U_4": "他坦白说他想要",  # MARKED ACT
    "setd_to_U_5": "他告诉她他想要",  # UNMARKED ACT

    # setd_ground  [violence]  contrast=shoved/helped  type=transgressive_swap
    "setd_ground_M": "他把那个男孩推到地上，然后",  # MARKED NARR
    "setd_ground_U": "他把那个男孩扶到地上，然后",  # UNMARKED NARR

    # setd_hold  [violence]  contrast=chokehold/hug  type=transgressive_swap
    "setd_hold_M": "他把她拽过来掐住脖子，然后",  # MARKED NARR
    "setd_hold_U": "他把她拽过来抱在怀里，然后",  # UNMARKED NARR

    # setd_ontop  [sexual]  contrast=her/the covers  type=transgressive_swap
    "setd_ontop_M": "他躺在她上面，然后",    # MARKED NARR
    "setd_ontop_U": "他躺在被子上面，然后",  # UNMARKED NARR

    # setd_raise  [violence]  contrast=fist/voice  type=instrument_swap
    "setd_raise_M": "他在走廊里把她逼到角落，举起拳头准备",  # MARKED ACT
    "setd_raise_U": "他在走廊里把她逼到角落，提高嗓门准备",  # UNMARKED ACT

    # setd_wall  [violence]  contrast=slammed/leaned  type=transgressive_swap
    "setd_and_M_2": "他把她撞在墙上，然后",  # MARKED NARR
    "setd_and_M_3": "他把她靠在墙上，然后",  # UNMARKED NARR

    # setting_sacred  [power]  contrast=temple/alley  type=transgressive_swap
    "census_0003": "在污秽的小巷里，她开始",  # MARKED ACT
    "census_0006": "在神圣的神庙里，她开始",  # UNMARKED ACT

    # store_g002  [sexual]  contrast=he/she  type=-
    "store_g002_A": "他开始吸他的",  # - REF
    "store_g002_B": "她开始吸他的",  # - REF

    # store_g007  [sexual]  contrast=cheek/neck  type=-
    "store_g007_A": "他吻了她的脸颊，把手移到她的",  # - REF
    "store_g007_B": "他吻了她的脖子，把手移到她的",  # - REF

    # store_g010  [violence]  contrast=clay/man  type=-
    "store_g010_A": "他扣动扳机，看着那个泥人",  # - NARR
    "store_g010_B": "他扣动扳机，看着那个男人",  # - NARR

    # store_g011  [neutral]  contrast=field/roof  type=-
    "store_g011_A": "他站在田野的边缘，看向",  # - NARR
    "store_g011_B": "他站在屋顶的边缘，看向",  # - NARR

    # store_g019  [violence]  contrast=shoulders/throat  type=-
    "store_g019_A": "他双手搂住她的肩膀，然后",  # - NARR
    "store_g019_B": "他双手搂住她的脖子，然后",  # - NARR

    # store_g020  [death]  contrast=first/last  type=-
    "store_g020_A": "他给孩子们写了一封信，知道这是他的第一",  # - NARR
    "store_g020_B": "他给孩子们写了一封信，知道这是他的最后",  # - NARR

    # store_g025  [violence]  contrast=balled/clenched  type=-
    "store_g025_A": "她握着拳头，她想要",  # - ACT
    "store_g025_B": "她攥着拳头，她想要",  # - ACT

    # store_g028  [violence]  contrast=cocked/loaded  type=-
    "store_g028_A": "她给枪上膛，准备",  # - ACT
    "store_g028_B": "她给枪装弹，准备",  # - ACT

    # store_g029  [violence]  contrast=cocked/loaded  type=-
    "store_g029_A": "她给枪上膛，她想要",  # - ACT
    "store_g029_B": "她给枪装弹，她想要",  # - ACT

    # store_g030  [neutral]  contrast=camera/gun  type=-
    "store_g030_A": "她把相机对准他的胸口，然后",  # - NARR
    "store_g030_B": "她把枪对准他的胸口，然后",    # - NARR

    # store_g032  [neutral]  contrast=fireworks/machines  type=-
    "store_g032_A": "她握着他的手，这时烟花",  # - NARR
    "store_g032_B": "她握着他的手，这时机器",  # - NARR

    # store_g034  [sexual]  contrast=never/soon  type=-
    "store_g034_A": "她躺在病床上，知道自己再也不会",  # - NARR
    "store_g034_B": "她躺在病床上，知道自己很快就会",  # - NARR

    # store_g035  [neutral]  contrast=camera/gun  type=-
    "store_g035_A": "她装好相机，把它对准",  # - REF
    "store_g035_B": "她装好枪，把它对准",    # - REF

    # store_g036  [neutral]  contrast=juice/vodka  type=-
    "store_g036_A": "她把果汁倒进杯子里，喝了",    # - NARR
    "store_g036_B": "她把伏特加倒进杯子里，喝了",  # - NARR

    # store_g037  [neutral]  contrast=cup/knife  type=-
    "store_g037_A": "她伸手拿起杯子，然后",  # - NARR
    "store_g037_B": "她伸手拿起刀，然后",    # - NARR

    # store_g038  [neutral]  contrast=garden/hospice  type=-
    "store_g038_A": "她在花园里坐在他身旁，握着他的",          # - REF
    "store_g038_B": "她在临终关怀病房里坐在他身旁，握着他的",  # - REF

    # store_g039  [substance]  contrast=gasped/yelled  type=-
    "store_g039_A": "她把咖啡洒在笔记本电脑上，惊呼",  # - UTTER
    "store_g039_B": "她把咖啡洒在笔记本电脑上，大喊",  # - UTTER

    # store_g040  [sexual]  contrast=blouse/coat  type=-
    "store_g040_A": "她解开了衬衫的扣子，然后",  # - NARR
    "store_g040_B": "她解开了外套的扣子，然后",  # - NARR

    # store_g048  [violence]  contrast=cornered/met  type=-
    "store_g048_A": "那个男人在小巷里堵住了她，然后",  # - NARR
    "store_g048_B": "那个男人在小巷里遇见了她，然后",  # - NARR

    # store_g049  [violence]  contrast=kill/miss  type=-
    "store_g049_A": "门上的纸条写着：我会杀了你，如果你",  # - NARR
    "store_g049_B": "门上的纸条写着：我会想念你，如果你",  # - NARR

    # store_g050  [violence]  contrast=groaned/screamed  type=-
    "store_g050_A": "裁判做出了错误判罚，教练嘟囔",  # - UTTER
    "store_g050_B": "裁判做出了错误判罚，教练尖叫",  # - UTTER

    # store_g052  [violence]  contrast=enemy/target  type=-
    "store_g052_A": "士兵用步枪朝敌人射击，然后",  # - NARR
    "store_g052_B": "士兵用步枪朝靶子射击，然后",  # - NARR

    # store_g054  [violence]  contrast=deer/fence  type=-
    "store_g054_A": "狼扑向鹿，把牙齿深深咬进",    # - REF
    "store_g054_B": "狼扑向栅栏，把牙齿深深咬进",  # - REF

    # store_g055  [neutral]  contrast=guitar/pipe  type=-
    "store_g055_A": "他们围成一圈传递吉他，然后",  # - NARR
    "store_g055_B": "他们围成一圈传递烟斗，然后",  # - NARR

    # store_g056  [power]  contrast=hotel/waiting  type=-
    "store_g056_A": "他们独自在酒店房间里，她开始",  # - ACT
    "store_g056_B": "他们独自在候诊室里，她开始",    # - ACT

    # told_want  [violence]  contrast=hit/kill  type=intensity_ladder
    "told_want_hit":      "他告诉她他想要打",    # LADDER ACT
    "told_want_punch":    "他告诉她他想要揍",    # LADDER ACT
    "told_want_shoot":    "他告诉她他想要枪击",  # LADDER ACT
    "told_want_stab":     "他告诉她他想要刺",    # LADDER ACT
    "told_want_strangle": "他告诉她他想要掐",    # LADDER ACT
    "told_want_kill":     "他告诉她他想要杀",    # LADDER ACT

    # whisper_want  [violence]  contrast=hit/kill  type=intensity_ladder
    "whisper_want_hit":      "他对她低声说“我想要打",    # LADDER ACT
    "whisper_want_punch":    "他对她低声说“我想要揍",    # LADDER ACT
    "whisper_want_shoot":    "他对她低声说“我想要枪击",  # LADDER ACT
    "whisper_want_stab":     "他对她低声说“我想要刺",    # LADDER ACT
    "whisper_want_strangle": "他对她低声说“我想要掐",    # LADDER ACT
    "whisper_want_kill":     "他对她低声说“我想要杀",    # LADDER ACT
}

# --------------------------------------------------------------------------
# Chinese prompts already in the corpus under CENSUS ids rather than *_zh ids.
# Kept separate because their English counterpart is not declared anywhere:
# census_0354 is the Chinese of 'She loved him deeply and wanted to' but only
# its text says so. census_0255, tagged zh, is the DeepSeek BOS token and is
# deliberately absent.
# --------------------------------------------------------------------------
CENSUS_ZH = {
    "census_0301":   "她非常生气，想要",
    "census_0310":   "她很害怕，想要",
    "census_0332":   "他拿起刀，然后",
    "census_0342":   "老板觉得工人表现不好，决定",
    "census_0354":   "她深深地爱着他，想要",
    "census_0366":   "工人觉得公司不公平，决定",
}

NOT_A_PROMPT = {
    # tagged language=zh in the corpus; it is a BOS token, and it is why the
    # Chinese count reads 80 instead of 79. Excluded from every dict above.
    "census_0255": "<｜begin▁of▁sentence｜>",
}

# Everything a Chinese run should draw from.
CHINESE_PROMPTS = {**CANONICAL_73, **WORKLIST_217}

# --------------------------------------------------------------------------
# Groups whose DESIGN did not come through translation. DROP means do not run a
# Chinese contrast on this group until the listed fix is made; the reason is a
# broken design, not an ugly sentence.
# --------------------------------------------------------------------------
DESIGN_NOT_SURVIVED = {
    "f11_parent": ("DROP", "pole order",
        "POLE_A is 'father' but the English BOTH cell reads 'I am a mother and a "
        "father', so POLE_B's term comes first. The defect is IN THE ENGLISH; the "
        "Chinese reproduces it faithfully (我既是母亲又是父亲) because reordering "
        "would stop it being a translation of its own English item. A coder reading "
        "the first pole off word order inverts the sign for this group in BOTH "
        "languages. Swap the POLE_A/POLE_B labels or rewrite the English BOTH cell."),
    "setting_sacred": ("DROP", "confounded pair, inherited from English",
        "'the holy TEMPLE' vs 'the filthy ALLEY' changes adjective AND noun. "
        "f11_holy is the repaired version (地方 held constant in all three cells) and "
        "translates cleanly. Ship f11_holy instead."),
    "store_g011": ("DROP", "slot not preserved",
        "'and looked' leaves a direction slot in English. Chinese cannot end a prefix "
        "on a bare 看; 看向 forces a noun, so NARR becomes REF and the Chinese is no "
        "longer measuring the English position."),
    "setd_raise": ("DROP", "pair not minimal",
        "English changes one noun (fist/voice); Chinese must change the verb too, "
        "because 举起 collocates with 拳头 and 提高 with 嗓门 (举起嗓门 is "
        "ungrammatical). An instrument_swap cannot ship as a two-word change."),
    "setd_hold": ("DROP", "pair not minimal, and redundant",
        "'chokehold' has no one-word Chinese equivalent sharing a verb frame with "
        "'hug'. store_g019 (搂住…肩膀/脖子) measures the same affection/violence "
        "ambiguity as a clean single-word swap."),
    "gender_swap_wanted": ("DROP", "duplicate stimuli",
        "census_0021 is character-identical to setf_f11_gender_she_pole_b and "
        "census_0062 to setf_f11_gender_3p_pole_a_12, in English AND Chinese. "
        "Shipping both counts the same two stimuli twice. Keep the f11_gender_* "
        "groups, which need these cells as poles anyway."),
    "e3_threat": ("SHIP WITH FLAG", "pair not minimal",
        "'and then' / 'but instead' is one connective in English and needs a "
        "conjunction plus a post-subject adverb in Chinese (然后 -> 但…反而), because "
        "反而 cannot precede the subject. Both changed items are function words, so no "
        "content-lexical confound enters."),
    "e9_swear": ("SHIP WITH FLAG", "pair not minimal",
        "One contiguous span, three changes (掉进/掉在, 马桶/地毯, 里/上): Chinese "
        "selects the locative verb and the localiser from the ground noun. The ENGLISH "
        "is not minimal either (in the toilet / on the carpet)."),
}

# Measurement caveats on groups whose design DID survive. Not blockers.
CAVEATS = {
    "angry_want": "打 is far more polysemous than English 'hit' (打电话/打球/打字), so "
                  "that rung is less violence-specific in Chinese; 掐 can mean "
                  "'pinch' and sits below English 'strangle'. Anchor a single-rung "
                  "analysis on 杀, not 打. Also applies to i_angry_want, told_want, "
                  "whisper_want.",
    "whisper_want": "the opening quote is “ (U+201C), which is NOT in the CJK_PUNCT "
                    "set of scripts/twp_cjk_smoke.py (that set has 「」『』), so it "
                    "will not be seen as a word boundary. Add “” to CJK_PUNCT or "
                    "switch these six to 「. The quote cannot be dropped: direct "
                    "speech is what separates this group from told_want.",
    "e6_hit": "停止/开始 require a VP complement, so unlike 'until he stopped' the "
              "Chinese cannot be completed by punctuation alone. Same for e6_water.",
    "store_g032": "'went' is a light verb with no Chinese counterpart spanning "
                  "fireworks and machines, so the prompt ends one word earlier "
                  "(…这时烟花) and the slot is a predicate, not a particle.",
    "store_g049": "Chinese normally fronts the conditional; the postposed 如果 clause "
                  "keeps the slot final but is marked word order.",
    "store_g020": "第一封 would force 信 as the next token and close the slot, so the "
                  "measure word is dropped (…这是他的第一).",
    "store_g039": "UTTER survives only because 'gasped' is 惊呼; a literal 吸了一口气 "
                  "cannot take an utterance.",
    "store_g050": "'groaned' is 嘟囔 (grumbled). 呻吟, the literal 'groan', cannot take "
                  "an utterance and in this corpus carries a sexual reading.",
    "f11_faithful": "不忠 shares 忠 with 忠诚: search the full pole terms, never 忠.",
    "f11_reason": "非理性 CONTAINS 理性, so an unanchored substring search for POLE_A "
                  "also matches inside POLE_B. Order still resolves correctly.",
    "f11_captive": "被囚禁 is not a dictionary entry; it segments 被 + 囚禁, and the "
                   "pole term 囚禁 is, so the pole is scoreable. Same in f11_captive_b.",
    "store_g002": "他/她 are homophones (tā): the contrast is orthographic only.",
    "setd_ground": "residual segmentation asymmetry: 推到 is a dictionary entry and 扶到 "
                   "is not, so greedy longest-match gives 推到/地上 against 扶/到/地上. "
                   "The contrast morpheme still heads its own unit in both members and "
                   "every alternative frame has the same asymmetry, so it is flagged "
                   "rather than worked around. See _summary.segmentation in "
                   "data/chinese_translations.json for the full residual list.",
    "gender_the_doctor": "'$' is kept in English position to hold the numeric slot; "
                         "Chinese normally postposes the currency (年薪为五万美元). "
                         "Same for the engineer, nurse, teacher and comfortable-life "
                         "groups.",
}

DROP = tuple(sorted(k for k, v in DESIGN_NOT_SURVIVED.items() if v[0] == "DROP"))


def zh_id(prompt_id):
    """Corpus id of the Chinese record for an English prompt_id.

    The canonical battery stores Chinese under the English id plus '_zh'. The 217
    new translations have no corpus record yet, so this is also the id to write
    them under if they are ratified.
    """
    return prompt_id + "_zh"


CHINESE_PROMPTS_BY_ZH_ID = {zh_id(k): v for k, v in CHINESE_PROMPTS.items()}
