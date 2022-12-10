################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization #diğerlerinden farklı olarak bu var hiperparametreyi dışsal olarak optimize etmeyi yapıcaz
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T #sayısal değişkenlerin betimsel istatistiklerine bakmak için yaparız bunu
df["Outcome"].value_counts() #bağımlı değişkenin sınıflarının dağılımlarına bakarız


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
#veri ön işlemede bi standartlaştırma yapmamız gerekiyor knn uzaklık temelli bir yöntemdir 
#uzaklık temelli yöntemlerde ve gradient descend temelli yöntemlerde değişkenlerin standart olması
#elde edilecek sonuçların ya daha hızlı ya da doğru olmasının sağlar
#yani daha başarılı olmamızı sağlar
#bu yüzden bağımsız değişkenleri bi standartlaştırma işlemine alıcaz. aşdk gibi
X_scaled = StandardScaler().fit_transform(X)
#yukarıdaki kodun çıktısı istediğimizi yapıyor ve bir numpy array'i dönüyor ama 
#sütun bilgisi vermiyor çıktı olan array bu yüzden sütunarı bağımsız değişkenlerin
#ilk halinden alıp yeni standartlaşmış olan bağımsız değişkenimize giriyoruz aşdk gibi
X = pd.DataFrame(X_scaled, columns=X.columns)


#makine öğrenmesi değişken mühendisliğine ve özellik mühendisliğine bağlıdır der coursera kurucusu andrew ng
#Data Preprocessing & Feature Engineering yani ortaya kayda değer bir şeyler koy burada kardeş
################################################
# 3. Modeling & Prediction
################################################
#           aslında bu fonksiyonun komşuluk sayısı hiperparametresi var ama şimdilik pas geçtik
knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)
#knn model'deki predict der ki hangi özelliği sorayım kardeş modele

#yalnız kardeş dikkat et modeli fit etmek yani train etmek ayrı bir süreç
#train edilen modeli kullanarak tahminde bulunmak ayrı bir süreçtir dikkatke

################################################
# 4. Model Evaluation
################################################
#bütün gözlem birimleri için knn modelle tahmin yapıyoruz aşağıdaki kod ile
# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
#bize auc ve roc hesabı için 1 mi 0 mı olduğu değil de 1 sınıfına ait olma olasılıkları lazım her birinin

y_prob = knn_model.predict_proba(X)[:, 1] #1. indexteki değerleri getir dedik

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74

# AUC
roc_auc_score(y, y_prob)
# 0.90

#bu zamana kadar yaptığımız işlemlerde modeli öğrendiği veriyle test ettik
#hadi train test olayına girelim
#hold out yöntemi mesela ya da cross validation'la hold out dezavatantajlarını bazı senaryolarda önleriz

#hadi 5 katlıyı kuralım
#cross validate der ki bana model nesneni ver, bağımsız değişkenlerini ver, bağımlı değ. ver
#kaç katlı cross validation istediğini ver, bir de kullanmak istediğin skor metriklerini ver
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
#cross_validate'in cros_val_score'dan farkı bir çok metriğe bakabilir

cv_results #kodunun çıktıları arasında score tıme ve fit time'da var bunlar tahmin etme süres
#ile alakalı modelin
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
#test_roc_auc'u hep farklı sonuç veriyor çünkü 4'üyle model kurdu 1'iyle test etti hep
#e böyle olunca bize de bunların ortalamasını alıp sonucu bulmak düşer.
cv_results['test_roc_auc'].mean()


# 0.73
# 0.59
# 0.78

#gördük ki değerlerimizde düşüş oldu
#ne anladık?Modeli kurduğumuz veriyi modelin performansını değerlendirmek için
#kullandığımızda yanlılık ortaya çıktı. Bu yanlılık ne getirir ne götürür noktasında
#dengesiz veri problemi var mı gibi bir soru işareti oluştuğunda mesela şu anki f1 skor tamam yok dedi rahatlaatı
#Veri setini cross validation yöntemiyle bölerek ayrı parçalarla model kurup
#diğer parçalarıyla test ettiğimizde tüm veriyle model kurmaya göre farklı bir sonuç aldık ve
#bu aldığımız yeni sonuçlar daha güvenilirdir.


#!!!!BU BAŞARI SONUÇLARI NASIL ARTTIRILABİLİR???!!!!!
# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme işleleri detaylandırılabilir
# 3. Özellik mühendisliği yeni değişkenler türetilebilir
# 4. İlgili algoritma için optimizasyonlar yapılabilir. 

knn_model.get_params()
#knn model'in dışsal bir hiperparametresi var ne bu komşuluk sayısı hiperparameteresi
#ve bu dışsal hiperparametre değiştirilebilirdir.
#parametre nedir? modelin veri içinden öğrendiği ağırlıklardır.
#hiperparametre nedir? kullanıcı tarafından girilmesi gereken ya da dışsal olan parametredir
#


################################################
# 5. Hyperparameter Optimization
################################################
#şimdi bu veriden öğrenilemeyen ve dışsal girilecek olan bu parametreni optimizasyonu için
#bir aralık giricez bir set olarak göndericez çöz gel kardeş diycez 
#en düşük hatayı veren hiperparametre değerlerini göster ben de ona göre optimize edeyim dicez

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model, #modelimizi verdik
                           knn_params, #değişkeni verdik
                           cv=5, #cross validation yapıyoz yine
                           n_jobs=-1, #işlemci performansını yüksek tutmak için
                           verbose=1).fit(X, y) #verbose rapor vercek bize
#bu grid ızgara kardeş her değişken için knn model kurup bakıcak tek tek
knn_gs_best.best_params_
#yukarıdaki kodla en iyi komşuluk hiperparametremizi bulduk kod 17 dedi

################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)
#                                 burada iki yıldız sebebi;
#                               key value değerlerini yazmamız gerek bu direkt ata dedi
#burada bir tane varken yazılabilirdi ama belki ileride onlarca olan bi modelimiz olucak
cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() #0.76
cv_results['test_f1'].mean() #0.61
cv_results['test_roc_auc'].mean() #0.81
#skorlar artmışşş good

random_user = X.sample(1)

knn_final.predict(random_user)











