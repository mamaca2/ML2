# XGBoost

პირველი მოდელი რაზეც დავიწყე მუშაობა არის XGBOOST.

## Data Loading
პირვლე რიგში გავაერთიანე train_identity და train_transaction id-ების საშუალებით, რათა ერთ სრულ data-ზე მემუშავა პირდაპირ.

## Cleaning
თავდაპირველად cleaning დავიწყე და პირველ რიგში წავშლე ის სვეტები რომელშიც 90% NaN იყო.
შემდეგ წავშალე ისეთი სვეტები რომელშიც ვარიაცია ძალიან ცოტა იყო. ასეთი სვეტები ვერ ეხმარება და არანაირ ინფორმაციას არ აწვდიან მოდელს ტრენინგში.
შემდეგ ტრენინგის დასაჩქარებლად მონაცემთა ზომა შევამცირე. ანუ თავიდან თუ float64-ში ეწერა, ახლა float32-ში გადავიყვანე.

## Feature Engineering

დროის feature-ები — TransactionDT არის წამებში გამოხატული elapsed time და არა რეალური timestamp.
მათემატიკური გარდაქმნებით გამოვიყვანე: transaction_day (წამები // 86400), transaction_hour ((წამები // 3600) % 24),
transaction_weekday (დღე % 7), transaction_month (დღე // 30).
ასევე შევქმენი is_night (საათი 0-6) და is_weekend (კვირის დღე 5-6).
ეს feature-ები მნიშვნელოვანია რადგან fraud ხშირად ღამით და შაბათ-კვირას ხდება, მაშინ როცა ბანკის გუნდი არ მუშაობს.

თანხის feature-ები — TransactionAmt-ს განაწილება ძლიერ right-skewed იყო, თითქმის ყველა მნიშვნელობა პირველ bin-ში იჩენდა თავს.
log1p გარდაქმნის შემდეგ განაწილება ბევრად სიმეტრიული გახდა და მოდელს გაუადვილდა pattern-ების სწავლა.
ასევე დავამატე cents remainder (თანხის ცენტების ნაწილი) და amt_is_round flag (მრგვალი თანხა თუ არა),
რადგან fraud-ის დროს ხშირად მრგვალი თანხები გამოიყენება.
bin-ები ($0-10, $10-50, $50-100, $100-500, $500-1000, $1000+) შევქმენი იმიტომ, რომ ზღვრული ეფექტები log-გარდაქმნაში პირდაპირ არ ჩანს, bin-ებში კი ნათლად ჩანს.

Frequency Encoding — card1, card2, addr1, P_emaildomain სვეტებისთვის გამოვთვალე თითოეული მნიშვნელობის სიხშირე
train set-ში და ეს სიხშირე ჩავანაცვლე ორიგინალ მნიშვნელობას. იშვიათი ბარათი ან მისამართი fraud-ის ინდიკატორი შეიძლება
იყოს. test set-ში უცნობი მნიშვნელობებისთვის 0 დავაყენე.

Group Aggregations — card1 და card2-ის მიხედვით დავაჯგუფე ტრანზაქციები და გამოვთვალე TransactionAmt-ის mean,
std და count. ეს მოდელს ეხმარება გაიგოს კონკრეტული ბარათის ჩვეული ქცევა. თუ ბარათი ჩვეულებრივ $30-ს ხარჯავს და
უცებ $3000 მოვიდა, std დიდი იქნება და ამას გავითვალისწინებთ.

Card ინტერაქცია — card1 და card2-ის კომბინაციის სიხშირე დავამატე. ცალ-ცალკე ეს სვეტები შეიძლება ხშირი იყოს,
მაგრამ მათი კომბინაცია "card1_card2" თითქმის უნიკალურად იდენტიფიცირებს კონკრეტულ ბარათს — კომბინაციის სიხშირე
უფრო ზუსტ სიგნალს იძლევა.

Email feature-ები — P_emaildomain და R_emaildomain-იდან გამოვყავი სუფიქსი (gmail.com → com) და შევქმენი email_match binary flag. fraud-ის დროს ხშირად გამგზავნისა და მიმღების email დომენი არ ემთხვევა.

Label Encoding — ყველა დარჩენილი object ტიპის სვეტი label encoding-ით გადავიყვანე რიცხვებში. train და test set-ი ერთად fit გავაკეთე რათა test-ში უცნობი კატეგორიები პრობლემა არ გამხდარიყო.

## Feature Selection

Feature selection-ისთვის სამი მიდგომა გავტესტე და 5-fold cross-validation-ით შევადარე ერთმანეთს:

Approach A — Correlation Filter — წავშალე ის სვეტები რომელთა |corr| > 0.95 სხვა სვეტთან. ძლიერ კორელირებული სვეტები ერთმანეთის დუბლიკატია და მოდელს ზედმეტ სიგნალს არ უმატებს.

Approach B — XGBoost Importance (Top-150) — XGBoost-ის gain-ზე დაყრდნობით ყველაზე მნიშვნელოვანი 150 feature შევარჩიე. gain გვიჩვენებს რამდენად ამცირებს კონკრეტული feature სხვადასხვა split-ებზე loss-ს.

Approach C — SelectFromModel (median threshold) — sklearn-ის SelectFromModel wrapper გამოვიყენე median threshold-ით — შეინახა ის feature-ები რომელთა importance median-ზე მეტია.

სამივე მიდგომის CV AUC შევადარე და საუკეთესო შედეგის მქონე approach-ი შეირჩა საბოლოო training-ისთვის. ყველა შედეგი MLflow-ზეა დალოგული XGBoost_Feature_Selection run-ში.

# RandomForest

მეორე მოდელი რაზეც დავიწყე მუშაობა არის RandomForest.

## Data Loading
პირველ რიგში გავაერთიანე train_identity და train_transaction id-ების საშუალებით, რათა ერთ სრულ data-ზე მემუშავა პირდაპირ.

## Cleaning
Cleaning ნაწილი იგივეა რაც XGBoost-ში.

## Feature Engineering
Feature Engineering ნაწილი იგივეა რაც XGBoost-ში. იგივე feature-ები შევქმენი.
## Feature Selection

Feature selection-ისთვის ოთხი მიდგომა გავტესტე. XGBoost-თან შედარებით დავამატე მეოთხე მიდგომა — RFE, რათა უფრო მეტი approach შემეფასებინა. ასევე correlation და CV სიჩქარის გასაზრდელად sampling გამოვიყენე — 30% train data-ზე ვათვლევინებდი CV-ს, რადგან RandomForest XGBoost-ზე ბევრად ნელია.

Approach A — Correlation Filter — წავშალე ის სვეტები რომელთა |corr| > 0.95 სხვა სვეტთან. correlation matrix-ის გამოსათვლელად 20% sample გამოვიყენე სიჩქარის გამო. ძლიერ კორელირებული სვეტები ერთმანეთის დუბლიკატია და მოდელს ზედმეტ სიგნალს არ უმატებს.

Approach B — RandomForest Importance (Top-150) — RandomForest-ის feature importance-ზე დაყრდნობით ყველაზე მნიშვნელოვანი 150 feature შევარჩიე. XGBoost-ისგან განსხვავებით აქ გამოიყენება impurity-based importance და არა gain.

Approach C — SelectFromModel (median threshold) — sklearn-ის SelectFromModel wrapper გამოვიყენე median threshold-ით — შეინახა ის feature-ები რომელთა importance median-ზე მეტია.

Approach D — RFE (top 100 features) — Recursive Feature Elimination გამოვიყენე DecisionTree-ზე დაყრდნობით. ეს მიდგომა iteratively შლის ყველაზე ნაკლებ მნიშვნელოვან feature-ებს სანამ 100 არ დარჩება. XGBoost-ში ეს მიდგომა არ გამომიყენებია, RandomForest-ში კი დავამატე რადგან feature-ების სიმრავლე მეტ პრობლემას ქმნის ხეებზე დაფუძნებულ ensemble-ებში.

ოთხივე მიდგომის CV AUC შევადარე 3-fold cross-validation-ით (XGBoost-ში 5-fold იყო — შევამცირე სიჩქარის გამო) და საუკეთესო შედეგის მქონე approach-ი შეირჩა საბოლოო training-ისთვის. ყველა შედეგი MLflow-ზეა დალოგული RandomForest_Feature_Selection run-ში.

## Training

### გამოცდილი მოდელები

RandomForest_depth3_trees10 - აშკარად underfitting-ის მაჩვენებელია. მხოლოდ 10 ხე და max_depth=3 არ ყეფნის რომ მოდელმა სრულად ისწავლოს pattern-ები.

RandomForest_depthNone_noreg — max_depth=None, min_samples_leaf=1, min_samples_split=2 ეს მონაცემები ცალსახად overfitting აჩვენებს. ხეები სრულად იზრდება და train AUC >> val AUC.

RandomForest_depth16_baseline — ყველაზე კარგი ექსპერიმენტი იყო: max_depth=16, min_samples_leaf=4, min_samples_split=10, class_weight='balanced'. class_weight='balanced' დავამატე რადგან dataset imbalanced-ია (~3.5% fraud) და მოდელს სჭირდება fraud კლასის მეტი ყურადღება.

### Hyperparameter ოპტიმიზაცია
Optuna TPESampler-ით 5 trial გაეშვა შემდეგი პარამეტრების სივრცეში:
n_estimators: 100-800, max_depth: 8-32, min_samples_leaf: 1-20, min_samples_split: 2-20, max_features: 0.1-1.0.
XGBoost-თან შედარებით trial-ების რაოდენობა შევამცირე 50-დან 5-მდე, რადგან RandomForest ბევრად ნელია და დრო არ გვქონდა.
თითოეული trial დაუყოვნებლივ ილოგებოდა MLflow-ზე nested run-ად, რათა progress რეალურ დროში ჩანდეს.

### საბოლოო მოდელი
საუკეთესო პარამეტრებით 3-fold OOF cross-validation გაეშვა მთელ train set-ზე.
საბოლოო მოდელი მთელ train set-ზე დაიტრენინგა და Pipeline-ად შეინახა MLflow Model Registry-ში სახელით RandomForest_Fraud_Pipeline.

საუკეთესეო შედეგი ამ მოდელიმა აჩვენა. თითქმის 0.90 აჩვენა private-ზე, ხოლო public-ზე 0.92

# AdaBoost

მესამე მოდელი რაზეც დავიწყე მუშაობა არის AdaBoost.

## Data Loading
ანალოგიურად ვიქცევი აქაც, როგორც წინა მოდელებზე.

## Cleaning
Cleaning ნაწილი იგივეა რაც XGBoost და RandomForest-ში.

## Feature Engineering
Feature Engineering ნაწილი იგივეა რაც წინა მოდელებში.

## Feature Selection

Feature selection-ისთვის იგივე ოთხი მიდგომა გამოვიყენე რაც RandomForest-ში, თუმცა რამდენიმე ცვლილება შევიტანე AdaBoost-ის სპეციფიკიდან გამომდინარე:

Approach A — Correlation Filter — იგივე მიდგომა: წავშალე ის სვეტები რომელთა |corr| > 0.95 სხვა სვეტთან.

Approach B — AdaBoost Importance (Top-100) — RandomForest-ისგან განსხვავებით Top-150-დან Top-100-ზე გადავედი. AdaBoost სუსტ base estimator-ებზე მუშაობს და ბევრი feature ხმაურს ქმნის მასში, ამიტომ ნაკლები feature უფრო კარგად მუშაობს.

Approach C — SelectFromModel (median threshold) — იგივე მიდგომა AdaBoost importance-ზე დაყრდნობით.

Approach D — RFE (top 80 features) — RandomForest-ისგან განსხვავებით 100-დან 80-ზე გადავედი და RFE estimator-ად DecisionTree გამოვიყენე RandomForest-ის მაგივრად. ეს ბევრად უფრო სწრაფია რადგან AdaBoost თავადაც DecisionTree-ზე მუშაობს და შესაბამისი feature-ები უნდა შეირჩეს.

ოთხივე მიდგომის CV AUC შევადარე 3-fold cross-validation-ით და საუკეთესო approach-ი შეირჩა საბოლოო training-ისთვის. ყველა შედეგი MLflow-ზეა დალოგული AdaBoost_Feature_Selection run-ში.

## Training

### გამოცდილი მოდელები

AdaBoost_n10_depth1 — underfitting, მხოლოდ 10 estimator და base_max_depth=1 (decision stump). მოდელი ვერ სწავლობს საკმარის pattern-ებს.

AdaBoost_n500_depth5 — overfitting დემონსტრაციისთვის შევქმენი ძლიერი მოდელი: 500 estimator და base_max_depth=5. წინა მოდელებისგან განსხვავებით AdaBoost-ში overfitting სხვაგვარად ვლინდება — learning_rate=1.0 და ღრმა ხეები მოდელს training data-ს დამახსოვრებისკენ უბიძგებს.

AdaBoost_n100_depth2_baseline — ბალანსი: 100 estimator, base_max_depth=2, learning_rate=0.1. learning_rate შევამცირე 1.0-დან 0.1-ზე რადგან მცირე learning rate უფრო კარგ გენერალიზაციას იძლევა — თითოეული ხე ნაკლებ წონას იღებს და ensemble ნელ-ნელა სწავლობს.

### Hyperparameter ოპტიმიზაცია
Optuna TPESampler-ით 20 trial გაეშვა. RandomForest-ისგან განსხვავებით trial-ების რაოდენობა 5-დან 20-მდე გავზარდე, რადგან AdaBoost ბევრად სწრაფია და მეტი trial-ის გაშვება შესაძლებელი იყო.
პარამეტრების სივრცე: n_estimators: 50-500, learning_rate: 0.01-1.0 (log scale), base_max_depth: 1-4.
base_max_depth 1-4-ით შევზღუდე რადგან AdaBoost-ში base estimator უნდა იყოს სუსტი — ღრმა ხეები overfitting-ს იწვევს.

### საბოლოო მოდელი
საუკეთესო პარამეტრებით 3-fold OOF cross-validation გაეშვა მთელ train set-ზე.
საბოლოო მოდელი მთელ train set-ზე დაიტრენინგა და Pipeline-ად შეინახა MLflow Model Registry-ში სახელით AdaBoost_Fraud_Pipeline.