# motscore (motor - Motor Diagnostic Confidence) - Motor test score. p=0.0
# tfcscore (motor? - Total Functional Capacity) - Total functional score. p=0.0
# ocularh (motor - Motor Diagnostic Confidence) - Group occular pursuit: horizontal. p=0.0
# sacinith (motor - Motor Diagnostic Confidence) - Group Saccade initiation: horizontal. p=0.0
# fingtapr (motor - Motor Diagnostic Confidence) - Finger tap right. p=0.0
# brady (motor - Motor Diagnostic Confidence) - Bradykinesia‐body. p=0.0
# dystlle (motor - Motor Diagnostic Confidence) - Group maximal dystonia. p=0.0002
# ---
# verfct6 (cogn - Cognitive Assessment) - Verbal fluency test: Total intrusion errors. p=0.0193
# sit2 (cogn - Cognitive Assessments) - Stroop Interference: total errors. p=0.0243
# swrt3 (cogn - Cognitive Assessments) - Stroop: Total self‐corrected errors. p=0.0249 (**SAME AS sit2**)
# scnt2 (cogn - Cognitive Assessments) - Stroop - total errors. p=0.0077
# ---
# pbas1sv (psych - Problem Behaviours Assessment) - Group depressed mood: Severity. p=0.0391.
# ---
# fascore (gen - UHDRS Functional Assessment Independence Scale) - Functional assmt score. p=0.0002
# indepscl (gen - UHDRS Functional Assessment Independence Scale) - Subject's independence in %. p=0.0
# p > 0.05: "mmsetotal", "irascore", "exfscore"
# ---
# GOOD P CURVE ON:
# "mmsetotal" (p-value > 0.05?)
# "sit2" (varies based on which other vars are in feature_list - presumably because different nas get dropped)
# "brady" (varies based on which other vars are in feature_list - presumably because different nas get dropped)


# Choose subj id + features + 2 covariates + hdcat
# features_list = ["subjid", "swrt1", "sit1", "mmsetotal", "cocfrq", "clb", "age", "isced", "hdcat"] # 5 fits w/work on mmse ["clb", "swrt1", "sit1"] mix of fits from below
# features_list = ["subjid", "scnt1", "swrt1", "sit1", "mmsetotal", "brady", "age", "isced", "hdcat"] # Was 4 fits - now 2
# features_list = ["subjid", "motscore", "swrt1", "sit1", "tfcscore", "mmsetotal", "age", "isced", "hdcat"] # 3 fits ["clb", "swrt1", "sit1"] mix of fits from below
# features_list = ["subjid", "scnt1", "swrt1", "sit1", "mmsetotal", "brady", "age", "isced", "hdcat"] # 2 fits ["swrt1", "sit1"] - COGNITIVE ONLY ("total correct" on 5 tests)
# features_list = ["subjid", "sit2", "mmsetotal", "tfcscore", "clb", "exfscore", "age", "isced", "hdcat"] # 2 fits ["sit2", "mmsetotal"]
# features_list = ["subjid", "pakfrq", "bar", "rigarml", "cocfrq", "herfrq",  "age", "isced", "hdcat"] # 2 fits ["cocfrq", "herfrq"] - T-TEST  >>ALL SUBSTANCES-BASED
# features_list = ["subjid", "motscore", "tfcscore", "dystlue", "prosupr", "brady", "age", "isced", "hdcat"] # 0 fits - T-TEST symptom tests  >>ALL MOTOR-BASED
# features_list = ["subjid", "tfcscore", "motscore", "age", "isced", "hdcat"] # 0 fits - T-TEST symptom tests  >>ALL MOTOR-BASED
# features_list = ["subjid", "tfcscore", "motscore", "mmsetotal", "irascore", "exfscore", "age", "isced", "hdcat"] # 1 fit ["tfcscore"] ORGINAL
# features_list = ["subjid", "hxbarfrq", "hxpak", "hxbar", "hxtrq", "hxpakfrq",  "age", "isced", "hdcat"] # 2 FITS ["clb" "clbfrq"] - RF TOP 5
