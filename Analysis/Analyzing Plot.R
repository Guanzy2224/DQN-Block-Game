# Read in data
setwd('D:/BOX AI')
dat = read.csv('Training Loss.csv')

# Plot earlier and infant episodes
par(mfrow = c(3,3))
for (i in (1:9)){
  vec = dat[i,]
  vec = as.numeric(vec)
  vec = na.omit(vec)
  plot(vec, type='l', main=paste("Episode", i), xlab="Step", ylab="Loss")
  
}

# Plot later mature episodes
par(mar=c(4,5,3,2),mfrow = c(3,3))
limit = 30
for (i in (1001:1009)){
  vec = dat[i,]
  vec = as.numeric((vec)^0.5)
  vec = na.omit(vec)
  outlier_y = vec[vec>limit]
  outlier_x = which(vec>limit)
  plot(vec, type='l', ylim = c(0,150), xlim = c(0,700),
       main=paste("Episode", i), xlab="Step", ylab=expression(sqrt("Loss")))
  if (length(outlier_x)>0){
    points(x = outlier_x, y = outlier_y, cex=3, col="red")
    text(x = outlier_x+50, y = outlier_y, labels = "Surprise", col="red")
  }
}
