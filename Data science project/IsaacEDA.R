# Isaac Waycott

par(mfrow = c(2, 2))
boxplot(test$model_year, main = "Model Year")
boxplot(test$milage, main = "Milage")
hist(test$model_year, main = "Model Year")
hist(test$milage, main = "Milage")


summary(test$milage)

summary(test$model_year)



milageOutliers = boxplot(test$milage, plot=FALSE)$out
modelYearOutliers = boxplot(test$model_year, plot=FALSE)$out


x = test
x = x[-which(x$model_year %in% modelYearOutliers),]
x = x[-which(x$milage %in% milageOutliers),]


summary(x$model_year)
 
summary(x$milage)



par(mfrow = c(2, 2))
boxplot(x$model_year, main = "Model Year")
boxplot(x$milage, main = "Milage")
hist(x$model_year, main = "Model Year")
hist(x$milage, main = "Milage")
