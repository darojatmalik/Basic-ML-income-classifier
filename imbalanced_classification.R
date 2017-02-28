library(ROSE)
library(rpart)

data(hacide)
str(hacide.train)
#cek table
table(hacide.train$cls)
#check classes distribution
prop.table(table(hacide.train$cls))

treeimb <- rpart(cls ~ ., data = hacide.train)
pred.treeimb <- predict(treeimb, newdata = hacide.test)
#looking for accuracy
accuracy.meas(hacide.test$cls, pred.treeimb[,2])
#check ROC curve model 
roc.curve(hacide.test$cls, pred.treeimb[,2], plotit = F)

#Make sampling data
#oversampling and balance data
data_balanced_over <- ovun.sample(cls ~ ., data = hacide.train, method = "over",N = 1960)$data
table(data_balanced_over$cls)
#undersampling
data_balanced_under <- ovun.sample(cls ~ ., data = hacide.train, method = "under", N = 40, seed = 1)$data
table(data_balanced_under$cls)
#oversampling and undersampling on this imbalanced data
data_balanced_both <- ovun.sample(cls ~ ., data = hacide.train, method = "both", p=0.5, N=1000, seed = 1)$data
table(data_balanced_both$cls)
#Synthetically with ROSE
data.rose <- ROSE(cls ~ ., data = hacide.train, seed = 1)$data
table(data.rose$cls)

#built decision tree modesl
tree.rose <- rpart(cls ~ ., data = data.rose)
tree.over <- rpart(cls ~ ., data = data_balanced_over)
tree.under <- rpart(cls ~ ., data = data_balanced_under)
tree.both <- rpart(cls ~ ., data = data_balanced_both)

#make prediction on unseen data
pred.tree.rose <- predict(tree.rose, newdata = hacide.test)
pred.tree.over <- predict(tree.over, newdata = hacide.test)
pred.tree.under <- predict(tree.under, newdata = hacide.test)
pred.tree.both <- predict(tree.both, newdata = hacide.test)

#evaluate the accuracy of respective prediction
#AUC ROSE
roc.curve(hacide.test$cls, pred.tree.rose[,2])
#AUC oversampling
roc.curve(hacide.test$cls, pred.tree.over[,2])
#AUC undersampling
roc.curve(hacide.test$cls, pred.tree.under[,2])
#AUC both
roc.curve(hacide.test$cls, pred.tree.both[,2])

#check model accuracy using holdout and bagging
ROSE.holdout <- ROSE.eval(cls ~ ., data = hacide.train, learner = rpart, method.assess = "holdout", extr.pred = function(obj)obj[,2], seed = 1)
ROSE.holdout

