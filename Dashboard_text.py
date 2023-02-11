def text(dashboardtext):
 
    if dashboardtext == 'home_text':
        return home_txt

    elif dashboardtext == 'home_addition_text':
        return home_addition_txt    

    elif dashboardtext == 'vacc_mor_text':
        return vacc_mor_txt
 
    elif dashboardtext == 'vacc_mor_conc_text':
        return vacc_mor_conc_txt

    elif dashboardtext == 'region_vacc_text':
        return region_vacc_txt
 
    elif dashboardtext == 'map_text':
        return map_txt

    elif dashboardtext == 'conc_text':
        return conclusion_txt    
  

home_txt = """The global outbreak of **Coronavirus disease 2019 (COVID-19)** pandemic has spread worldwide, 
affecting most people around the world. Due to the long lasting effect of COVID-19 on the people's mental 
health and well-being, rulers of countries and territories cautioned the public to take responsive care
[1](https://journals.sagepub.com/doi/pdf/10.1177/2347631120983481)[2](https://www.cambridge.org/core/journals/the-british-journal-of-psychiatry/article/mental-health-and-wellbeing-during-the-covid19-pandemic-longitudinal-analyses-of-adults-in-the-uk-covid19-mental-health-wellbeing-study/F7321CBF45C749C788256CFE6964B00C).<br>
Yet, it is shown in the diagram below that the total number of __COVID__ cases is having an upward trend. To 
stop this chronic disease, other methods has been taken on.<br> In this study we are going to investigate 
how death preventer **"Vaccination"** has been to people in the UK"""

home_addition_txt = """ While inspecting the data available, we realized there is no information about 
vaccination in __Northern Ireland__ and __Wales__.<br> The other data for these two regions were not subject 
related, therefore they are completely dropped from this investigation"""

vacc_mor_txt = """The below chart shows total vaccination and mortality for each region."""

vacc_mor_conc_txt = """We know, from the chart, that the fatal effect of __COVID__ in all regions is much 
less than the total number of people vaccinated.<br> but this doesn't answer our question. Consequently, 
I would like to attract your attention to the charts in the next tab """

region_vacc_txt = """Here we see the graph representing the change in death numbers respective to vaccination 
rate.<br> To have better view point, Regression line has been drawn on the scatters"""
  
map_txt = """Below you will see a geographic map regarding to vaccination and mortality rate.
<br>"""

conclusion_txt = """Based on horizontal regression lines we sawin the other tab, we can believe that our 
hypothesis is rejected and there is no proof that vaccination is in direct relation with mortality rate. 
further studies need to be done to determine the reliability of this hypothesis."""
