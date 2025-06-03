library(rinat)
library(lubridate)
library(readr)
library(dplyr)
library(tidyr)
library(anytime)
library(mgcv)
library(sjPlot)
library(ggplot2)
library(chillR)
library(suncalc)

#download observations from each county
resultswake = get_inat_obs(
  taxon_name = "Anolis carolinensis",
  place_id = 1844,
  quality = "research",
  maxresults = 10000
)

write.csv(resultswake, "E://a_carol_data/wake_raleigh/wake_list.csv")

download_images <- function(dat,size="medium",outpath="E://a_carol_data/wake_raleigh/all_images/"){
  for (i in 1:dim(dat)[1]){
    iurl <- dat$image_url[i]
    iurl <- gsub("medium",size,iurl)
    iname <- paste(outpath,dat$id[i]," ",dat$scientific_name[i]," ",dat$observed_on[i],".jpg",sep="")
    print(iname)
    tryCatch(
      {download.file(iurl,iname, mode = 'wb')},
      error = function(err) {print(paste("MY_ERROR:  ",err))}
    )
    Sys.sleep(2)
  }
}

download_images(resultswake)

#merge data with color output from pipeline for each county
wake_colorout <- merge(wake_list, wake_colorout, by = 'id', all.y = TRUE )
wake_colorout$county <- "wake"

#merge all counties for a final, labeled data set
finaldata <- rbind(travis_colorout, richland_colorout, fulton_colorout, harris_colorout, hillsborough_colorout, 
                   alachua_colorout, horry_colorout, jefferson_colorout, leon_colorout, mecklenburg_colorout,
                   miamidade_colorout, newhanover_colorout, nueces_colorout, orange_colorout, orleans_colorout,
                   dallas_colorout, chatham_colorout, wake_colorout, bexar_colorout, broward_colorout)

#add time information based on observation date
finaldata$weeks <- week(ymd(finaldatat$observed_on))
finaldata$yday <- yday(finaldata$observed_on)
finaldata$year <- year(finaldata$observed_on)

#prep data for week-based analysis
summary_data_final_nocounty <- finaldata %>%
  group_by(weeks, color) %>%
  tally() %>%
  View()

summary_data_final_nocounty$percent_brown <- summary_data_final_nocounty$brown / (summary_data_final_nocounty$brown + summary_data_final_nocounty$green)
summary_data_final_nocounty$n <- summary_data_final_nocounty$brown + summary_data_final_nocounty$green

#run cyclical GAM
gam1 <- gam(percent_brown ~ s(weeks, bs= "cc", k = 3), weights = n, 
            family=binomial, data=summary_data_final_nocounty)
View(gam1)

#plot bar and GAM
plot(summary_data_final_nocounty$weeks, summary_data_final_nocounty$percent_brown, 
     main = "Proportion of Anoles Presenting Brown", xlab = "Week", ylab = "Proportion")
plot_model(gam1, type="pred", axis.title = c("Week","Percent Brown"), terms = c("weeks"),
           title = "Predicted Proportions of Anoles Presenting Brown per Week")

#fetch the min-max temps for each observation
data_w_temp <- merge(finaldata, climatic_data_for_serena, by = c("year","county","yday"))

#fetch the sunrise and sunset data for each observation
sunlightdata$date <- date(sunlightdata$date)
sunlightdata <- data.frame(date = data_w_temp$date,lat = data_w_temp$latitude.y,
                           lon = data_w_temp$longitude.y)
sunlightdatafull <- getSunlightTimes(data = sunlightdata,
                                     keep = c("sunrise", "sunriseEnd", "sunset", "sunsetStart"))

#I can't remember if you redid getting the sunlight data, or if I sent you the data with sunlight already added

#jacobs stuff

#set brown and green to 0,1 and round temperatures
serena_data_id <- serena_data %>%
  mutate(colorid = if_else(color == "green", 1, 0))
serena_data_id$rounded <- round(serena_data_id$hour_temperature, 1)

#s curve stuff we still need to do will go here

#make subsets for summer and non-summer data
#summer excluded
excludedb4 <- filter(serena_data_id, month < 4 | month > 8)
#only summer
notexcluded <- filter(serena_data_id, month > 4 | month < 8)

#run glm for each group, this does not require summarizing or filtering and seems to be a better fit
glm_all <- glm(colorid ~ hour_temperature + local_hour, family = "binomial", data = serena_data_id)
glm_noex <- glm(colorid ~ hour_temperature + local_hour, family = "binomial", data = notexcluded)
glm_ex <- glm(colorid ~ hour_temperature + local_hour, family = "binomial", data = excludedb4)

#plot models
plot_model(m_all, type = "pred", grid.breaks = c(5,0.2), terms = c("hour_temperature [all]"), 
           title = "Predicted Probabilities of Anoles Presenting Brown", axis.title = c("Temperature (C)","Percent Green"))
plot_model(m_noex, type = "pred", grid.breaks = c(5,0.2), terms = c("hour_temperature [all]"), 
           title = "Predicted Probabilities of Anoles Presenting Brown for Months May through September", axis.title = c("Temperature (C)","Percent Green"))
plot_model(m_ex, type = "pred", grid.breaks = c(5,0.2), terms = c("hour_temperature [all]"), 
           title = "Predicted Probabilities of Anoles Presenting Brown for Months November through April", axis.title = c("Temperature (C)","Percent Green"))

#summarize data for each state and get proportions... I should turn this into a function

#all
summary_data_final_wtemp <- serena_data_id %>%
  group_by(rounded, color) %>%
  tally() %>%
  spread(key = color, value = n, fill = 0)
View()
summary_data_final_wtemp$percent_brown <- summary_data_final_wtemp$brown / (summary_data_final_wtemp$brown + summary_data_final_wtempgreen)
summary_data_final_wtemp$n <- summary_data_final_wtemp$brown + summary_data_final_wtemp$green

#summer excluded
summary_data_final_wtempex <- excludedb4 %>%
  group_by(rounded, color) %>%
  tally() %>%
  spread(key = color, value = n, fill = 0)
View()
summary_data_final_wtempex$percent_brown <- summary_data_final_wtempex$brown / (summary_data_final_wtempex$brown + summary_data_final_wtempexgreen)
summary_data_final_wtempex$n <- summary_data_final_wtempex$brown + summary_data_final_wtempex$green

#summer
summary_data_final_wtempnoex <- nonexcluded %>%
  group_by(rounded, color) %>%
  tally() %>%
  spread(key = color, value = n, fill = 0)
View()
summary_data_final_wtempnoex$percent_brown <- summary_data_final_wtempnoex$brown / (summary_data_final_wtempnoex$brown + summary_data_final_wtempnoexgreen)
summary_data_final_wtempnoex$n <- summary_data_final_wtempnoex$brown + summary_data_final_wtempnoex$green

#optionally filter for temperatures where n > 10... we've ran it with and without we should probably discuss
summary_data_temp <- filter(summary_data_final_wtemp, n > 10)
summary_data_tempnoex <- filter(summary_data_final_wtempnoex, n > 10)
summary_data_tempex <- filter(summary_data_final_wtempex, n > 10)

#generate lm
lm_all <- lm(percent_brown ~ rounded, data = summary_data_temp)
lm_noex <- lm(percent_brown ~ rounded, data = summary_data_wtempnoex)
lm_ex <- lm(percent_brown ~ rounded, data = summary_data_wtempex)

#plot lm
plot_model(lm_all, type = "pred", grid.breaks = c(5,0.2), terms = c("rounded"), 
           title = "Predicted Probabilities of Anoles Presenting Brown", axis.title = c("Temperature (C)","Percent Brown"))
plot_model(lm_noex, type = "pred", grid.breaks = c(5,0.2), terms = c("rounded"), 
           title = "Predicted Probabilities of Anoles Presenting Brown for Months May through September", axis.title = c("Temperature (C)","Percent Brown"))
plot_model(lm_ex, type = "pred", grid.breaks = c(5,0.2), terms = c("rounded"), 
           title = "Predicted Probabilities of Anoles Presenting Brown for Months November through April", axis.title = c("Temperature (C)","Percent Brown"))
