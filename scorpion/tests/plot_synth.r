library(RPostgreSQL)
library(ggplot2)

drv = dbDriver('PostgreSQL')
con = dbConnect(drv, dbname='sigmod')
#data = dbGetQuery(con, "SELECT distinct expid,  cost, regexp_replace(rule, E'^\\-?[\\\\d\\.]+\\\\s+[\\\\d\\.]+\\\\s*', '') as rule from results")
data = dbGetQuery(con, "SELECT distinct dataset, r.expid, md.uo, md.volperc, c, cost, r.klass, md.ndim, s.notes from results as r, stats as s, dataset_metadata as md where r.expid > 47 and r.id = s.resultid and r.dataset = md.tablename ")

#data[data$klass=='MR'&data$uo==30&data$notes=='oboxes'&data$ndim==4,]

head(data)
p = ggplot(data, aes(x=ndim, y=cost, fill=klass, color=klass, group=klass)) 
p = p + geom_line() + geom_point() + facet_grid(uo~volperc)
p = p + theme(legend.position='bottom') + theme_bw()
ggsave('./costs.pdf', plot=p, scale=2)



q = paste("SELECT bestrules.dataset, bestrules.klass, bestrules.volperc, bestrules.uo, bestrules.ndim, bestrules.expid, bestrules.c, bestrules.score, s.notes, s.f1, s.prec, s.recall",
"FROM stats as s, (",
"  SELECT  r2.klass, r2.dataset, ndim, uo, volperc, r2.expid, r2.id, r2.c, r2.score",
"  FROM (",
"     SELECT r.expid, r.c, md.ndim, md.uo, md.volperc, max(r.score) as score ",
"     FROM results as r, dataset_metadata as md ",
"     WHERE md.tablename = r.dataset",
"     group by r.expid, r.c, md.ndim, md.uo, md.volperc",
"   ) as keys, results as r2",
"  WHERE r2.expid > 47 and r2.expid = keys.expid and r2.c = keys.c and r2.score = keys.score) as bestrules",
"WHERE s.resultid = bestrules.id ", sep=' ')



data = dbGetQuery(con, q)
head(data)

score = data
score$v = data$score
score$v = score$v/max(score$v)
score$measure = "Score"
f1 = data
f1$v = data$f1
f1$measure = "F1"
prec = data
prec$v = data$prec
prec$measure = "Precision"
recall = data
recall$v = data$recall
recall$measure = "Recall"
merged = rbind(score, f1, prec, recall)



p = ggplot(merged[merged$uo==30&merged$volperc==.5,], aes(x=c, y=v, color=paste(notes, klass), group=paste(notes, klass, expid))) + geom_point() + geom_line() + facet_grid(measure~ndim)
p = p + theme_bw()
ggsave('./statshard.pdf', plot=p, width=10, height=6, scale=2)

p = ggplot(merged[merged$uo==80&merged$volperc==.5,], aes(x=c, y=v, color=paste(notes, klass), group=paste(notes, klass, expid)))
p = p + geom_line() + geom_point() + facet_grid(measure~ndim)
p = p + theme_bw()
ggsave('./statseasy.pdf', plot=p, width=10, height=6, scale=2)


p = ggplot(data, aes(x=c, y=f1, color=paste(klass), group=paste(klass))) 
p = p + geom_line() + facet_grid(volperc+uo~ndim+notes)
p = p + theme_bw()
ggsave('./f1.pdf', plot=p, width=10, height=6, scale=2)

p = ggplot(data, aes(x=c, y=prec, color=paste(klass), group=paste(klass))) 
p = p + geom_line() + facet_grid(volperc+uo~ndim+notes)
p = p + theme_bw()
ggsave('./perc.pdf', plot=p, width=10, height=6, scale=2)


p = ggplot(data, aes(x=c, y=recall, color=paste(klass), group=paste(klass))) 
p = p + geom_line() + facet_grid(volperc+uo~ndim+notes)
p = p + theme_bw()
ggsave('./recall.pdf', plot=p, width=10, height=6, scale=2)




