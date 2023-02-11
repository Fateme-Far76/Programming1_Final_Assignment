#!/usr/bin/env python
# coding: utf-8

# # Introduction
# "In December 2019, a novel coronavirus, now named as SARS-CoV-2, caused a series of acute atypical respiratory diseases in Wuhan, Hubei Province, China. The disease caused by this virus was termed COVID-19. The virus is transmittable between humans and has caused pandemic worldwide. The number of death tolls continues to rise and a large number of countries have been forced to do social distancing and lockdown"[[1]](https://www.sciencedirect.com/science/article/pii/S152166162030262X). <br>
# Here we are going to do a research on how effective vaccination has been with respect to death rate

# Importing the necessary libraries
# 

# In[1]:


import panel as pn
import yaml
import pandas as pd
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, FactorRange, HoverTool 
from bokeh.models import DatetimeTickFormatter 
import folium
import json
import plotly.express as px
from bokeh.models import NumeralTickFormatter
  
output_notebook()
 


# # Loading the data
# This covid data was sourced from [here](https://www.kaggle.com/datasets/albertovidalrod/uk-daily-covid-data-countries-and-regions) and population of counties were driven from the xls file [here](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland) <br>
# The covid files have 16 columns, only 6 of them has been selected for this study.<br> 
# In the other file there are a lot data about all countries, but only population of UK regions
# Here is an explaination of each column:<br>
# 
#     "date": date in YYYY-MM-DD format 
#     "area name": name of area covered in the file (region or nation name)
#     "cum cases": cumulative cases 
#     "daily cases": new cases 
#     "new deaths 28days": new deaths within 28 days of a positive test 
#     "cum deaths 28days": cumulative deaths within 28 days of a positive test 
#     "cum vaccinations": cumulative people vaccinated booster or third dose 
# 
# 

# In[2]:


# Opening the config.yaml file and to extract filepath from the file
with open ('config.yaml', 'r') as file_adress:
    config = yaml.safe_load(file_adress)

# reading each file
Englandfile = config['Englandpath']
EMidlandsfile = config['EMidlandspath']
Eastfile = config['Eastpath']
Londonfile = config['Londonpath']
NorthWestfile = config['NorthWestpath']
SouthWestfile = config['SouthWestpath']
YandHfile = config['Y&Hpath']
NIrelandfile = config['NIrelandpath']
Scotlandfile = config['Scotlandpath']
Walesfile = config['Walespath']
 
# making a dataframe of each file
England_df = pd.read_csv(Englandfile)
EMidlands_df = pd.read_csv(EMidlandsfile)
East_df = pd.read_csv(Eastfile)
London_df = pd.read_csv(Londonfile)
NorthWest_df = pd.read_csv(NorthWestfile)
SouthWest_df = pd.read_csv(SouthWestfile)
YandH_df = pd.read_csv(YandHfile)
NIreland_df = pd.read_csv(NIrelandfile)
Scotland_df = pd.read_csv(Scotlandfile)
Wales_df = pd.read_csv(Walesfile)


# # Data inspection and processing

# In[3]:


#England_df.info()
#EMidlands_df.info()
#East_df.info()
#London_df.info()
#NorthWest_df.info()
#SouthWest_df.info()
#YandH_df.info()
#NIreland_df.info()
#Scotland_df.info()
Wales_df.info()


# from the code above we realized that all files have the same columns

# In[4]:


# making a list of dataframes for further use
df_namelist = [England_df, EMidlands_df, East_df, London_df, NorthWest_df, SouthWest_df,
            YandH_df, NIreland_df, Scotland_df, Wales_df]

#changin the name of this region to be the same in all files
East_df['area_name'] = 'East'

# creating an empty dataframe
UK_df = pd.DataFrame()
 
# filtering the columns we need and merging all data from all regions to have 1 big dataframe. 
# with this method the original data stays intact and we only glue the rows of one dataframe to another 
for df in df_namelist:

    new_df = df[['date', 'area_name', 'cum_vaccinations' , 'daily_cases' , 'cum_cases', 'new_deaths_28days','cum_deaths_28days']].copy()
    UK_df = pd.concat([UK_df, new_df], ignore_index = True, sort = False)

#checking data types and null counts 
UK_df.info()


# In[5]:


# converting to date
UK_df['date'] = pd.to_datetime(UK_df['date'])

# this data is discrete so it doesn't need to be float
UK_df['cum_vaccinations'] = UK_df['cum_vaccinations'] .astype('Int64') 
UK_df['daily_cases'] = UK_df['daily_cases'].astype('Int64')
UK_df['new_deaths_28days'] = UK_df['new_deaths_28days'].astype('Int64')
UK_df['cum_cases'] = UK_df['cum_cases'].astype('Int64')
UK_df['cum_deaths_28days'] = UK_df['cum_deaths_28days'].astype('Int64') 
  


# In[6]:


# making sure that there aren't any other area mixed up here  
UK_df.area_name.unique()


# In[7]:


# reading population file
populationfile = config['poppath']
original_population_df = pd.read_excel(populationfile, sheet_name = "MYE2 - Persons", skiprows = range(7)) 

# list of region names
regionlist = UK_df['area_name'].unique( ).tolist()

# replacing names in population dataframe with names in 
for region in regionlist:
    original_population_df['Name'] = original_population_df['Name'].replace([region.upper()], region)
 
# making a data frame from our desired regions 
population_df  = original_population_df[original_population_df['Name'].isin(regionlist)].reset_index(drop = True)

# check if all 10 regions are found
population_df


# In[8]:


# making a copy of region name and their respecitve population columns to work on it
population_df = population_df[[ 'Name','All ages']].copy()

# renaming and saving column name
population_df.rename(columns = {"All ages": "Population"}, inplace=True)
 
# renaming area column name for better call up and merging
UK_df.rename(columns = {"area_name": "Name"}, inplace=True)

# adding population column to the dataframe
UK_merg_df = pd.merge(UK_df, population_df, left_on = 'Name', right_on = 'Name') 

# checking the info to see the added column's type 
UK_merg_df.info()


# In[9]:


#changin the name of this region to normal
UK_merg_df['Name'] = UK_merg_df['Name'] .replace(['East'],[ 'East of England'])


# In[10]:


# deleting 'NA' values from columns to prepare for plotting
UK_merg_df.dropna(subset = 'cum_vaccinations', inplace = True)
UK_merg_df.dropna(subset = 'new_deaths_28days', inplace = True)
UK_merg_df.info()   


# Apparently there are a lot of missing data, lets check what's going on

# In[11]:


#printing length of data of each region
regions = UK_merg_df.Name.unique()
for region in regions:
    len_region = len(UK_merg_df[UK_merg_df.Name == region])
    print(f'The length of {region} data is {len_region}')


# As we see 2 regions are completely droped from our dataframe, 
# it means that there is no available information about vaccination in Northern Ireland and Wales.
# Although it's not nice to conclude without all regions considered, keeping uninformative data doesn't help this research question 
# 

# # Hypothesis check

# we assume that vaccination result in less mortality.
# first lets take a look at the spreading trend of this virus in the UK 

# In[12]:


#ploting the graph for the trend of new cases

p = figure(x_axis_type = 'datetime') 
 
for region, color in zip(regions, px.colors.sequential.Plasma):
    df = UK_merg_df[UK_merg_df.Name == region]
    p.line(x = df['date'], y =  df['cum_cases'], line_width = 2, color = color, alpha = 0.8, legend_label = region ) 
 
p.xaxis.axis_label = 'Date' 
p.yaxis.axis_label = 'Total number of cases'
p.grid.grid_line_color = "white"
p.legend.click_policy = "hide"
p.title.text_color = 'midnightblue'
p.yaxis[0].formatter = NumeralTickFormatter(format = "000,000") 
p.xaxis.formatter=DatetimeTickFormatter(months="%b %y")  
p.legend.location = 'top_left'
 
trend_plot = p
show(trend_plot)


# As we can see the trendline is upward in all regions, and England is showing lot more cases than the others!
# lets check the total number of cases respective to their region population to see if it is because of the population or not

# In[13]:


# making some random colors ( copied from bokeh website :D ) 
x = np.random.random(size = 8) *100
y = np.random.random(size = 8) *100
colors = np.array([ [r, g, 150] for r, g in zip(50 + 2*x, 30 + 2*y) ], dtype="uint8") 

# making a new dataframe for ploting
compare_df  = UK_merg_df.groupby("Name")[ 'Population',"cum_cases"].max().reset_index()

# data we wanna plot
source = ColumnDataSource(data=dict(x = compare_df.cum_cases, 
        y = compare_df.Population, 
        name = compare_df.Name, 
        size = compare_df.cum_cases/400000,
        colors = colors)) 

# filling data we want to show when hovering        
hover = HoverTool(tooltips = [ 
    ("Name", '@name'),
    ("Total Cases", "@x"),
    ('Population', '@y') ]) 

# plot
p = figure(sizing_mode = "stretch_width", 
           plot_height = 500, 
           tools = [hover])

p.circle('x', 'y', size = 'size', source=source,
         fill_color = 'colors')

p.xaxis.formatter = NumeralTickFormatter(format = "000,000")  
p.yaxis.formatter = NumeralTickFormatter(format = "000,000")  
p.xaxis.axis_label = 'Total Cases' 
p.yaxis.axis_label = 'Population'

population_case_plot = p 
show(population_case_plot)


# Yep, we can see that England is the most populated among them. The more the population the more + cases. makes sense :)

# now lets check the amount vaccinated and amount dead, and see if we can prove our hypothesis

# In[14]:


region_max_vacc = []
region_max_death = []

for region in regions:

        # calculating the total vaccination percentage in each region 
        max_vacc = UK_merg_df[UK_merg_df.Name == region]['cum_vaccinations'].max()
        max_vacc_ratio = max_vacc/(UK_merg_df[UK_merg_df.Name == region]['Population'].max()) * 100
        region_max_vacc.append(max_vacc_ratio)

        # calculating the total death percentage in each region 
        max_death = UK_merg_df[UK_merg_df.Name == region]['cum_deaths_28days'].max()
        max_death_ratio = max_death/(UK_merg_df[UK_merg_df.Name == region]['Population'].max()) * 100
        region_max_death.append(max_death_ratio)

xaxis = ['Total vaccination', 'Total mortality'] 

# a dictionary of the data wewant to plot
data = {'Total vaccination'   : region_max_vacc,
        'Total mortality'   : region_max_death }

# this creates a list of (region,total vaccinatin)set and (region,total death) set
x = [ (region, dataa) for region in list(regions) for dataa in xaxis ]

# then we need to make some kind of list (in this case it's not a list) of the total vaccination percentage and total death percentage of regions
counts = sum(zip(data['Total vaccination'], data['Total mortality'] ), ( )) 

# specifying the source for plotting
source = ColumnDataSource(data = dict(x = x, counts = counts))

# unpacking sets in x list, turning off the toolbar for the plot and finally drawing the plot 
p = figure(x_range = FactorRange(*x), 
           plot_height = 600, 
           plot_width = 950, 
           toolbar_location = None, 
           title = "Total vaccination vs mortality per region")

p.vbar(x = 'x', 
       top = 'counts', 
       width = 0.9, 
       source = source)

 
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.y_range.start = 0
p.yaxis.axis_label = 'Percentage'
p.xaxis.major_label_orientation = 1.3 
p.title.text_color = 'midnightblue'
vacc_mortality_plot = p
show(vacc_mortality_plot) 


# now lets check the regression line , according to our hypotesis it should be downward

# In[15]:


# adding an column filled with 0 to calculate vaccination ratio
UK_merg_df = UK_merg_df.assign(vacc_ratio = [0]*len(UK_merg_df))
 
def vacc_ratio_fill(row):
    '''
    Calculates vaccination ratio for each row and returns the row
    '''
    row.vacc_ratio = row.cum_vaccinations / row.Population * 100 
    return row

# filling vaccinati0on ratio column with the correct numbers 
UK_merg_df = UK_merg_df.apply(vacc_ratio_fill, axis=1)

# checking the type of new column
UK_merg_df.vacc_ratio.dtype 


# In[16]:


# casting the new column to int
 
UK_merg_df['vacc_ratio'] = UK_merg_df['vacc_ratio'].astype('str').str.extract(r'([0-9]+.[0-9]+)').astype('float')
  


# In[17]:


# an empty list for plots to be added
vac_death_scatter_plot = []

# drawing a scatter plot for each region 
for region in regions:
     
    region_df = UK_merg_df[UK_merg_df.Name == region] 

    p = figure(plot_width = 500, 
                plot_height = 400, 
                title = region, 
                toolbar_location = None)  

    p.circle(region_df.vacc_ratio,
             region_df.new_deaths_28days, 
             size = 13, 
             line_color = "navy", 
             fill_color = "skyblue", 
             fill_alpha = 0.4)

    # doing a linear curve fitting to check the hypothesis          
    coef = np.polyfit(region_df.vacc_ratio, region_df.new_deaths_28days, deg = 1)
    curve_fit = np.poly1d(coef)
    y = curve_fit(region_df.vacc_ratio) 
    p.line(x = region_df.vacc_ratio, y = y, line_color = 'red', line_width = 2, legend_label = 'Linear Regression')
    p.xaxis.axis_label = 'Vaccination rate (%)'
    p.yaxis.axis_label = 'New deaths' 
    vac_death_scatter_plot.append(p) 
 


# In[18]:


from scipy.stats import pearsonr 
stat, p = pearsonr(UK_merg_df['cum_vaccinations'], UK_merg_df['cum_deaths_28days'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


# In[19]:


UK_merg_df['cum_vaccinations'].corr(UK_merg_df['cum_deaths_28days'])


# these show that there is some relation but none can prove vaccination is effective for stoping people from dying out of COVID

# GIS

# In[114]:



# loading geojson file 
UKgeofile = config['UK_geojson']
geojson_file = open(UKgeofile)
geojson_data = json.load(geojson_file)
geojson_file.close()
 
# creating a global map showing UK
UKmap = folium.Map(location = [55.677584411089526,-3.4936523437500004], zoom_start = 5)

# regions border style
borders = {
    "color" : "blue", 
    "fill" : False, 
    "weight" : 4}

folium.GeoJson(data = geojson_data, 
    name = 'borders', 
    style_function = lambda x : borders).add_to(UKmap)

 
max_vacc = UK_merg_df["vacc_ratio"].max() 
max_death = UK_merg_df["cum_deaths_28days"].max()
folium.Choropleth(
    geo_data = UKgeofile,
    name = 'Vaccination',
    data = UK_merg_df,
    columns = ['Name', 'vacc_ratio'],
    key_on = 'feature.properties.RGN21NM',
    fill_color = 'PuBu',
    fill_opacity = 0.8,
    line_opacity = 0.2,
    legend_name = 'Vaccination Rate',
    bins = [0, 10, 15, 20, 30, 40, 50, max_vacc],
    nan_fill_color = 'white',
    show = False
).add_to(UKmap) 
'''
folium.Choropleth(
    geo_data = UKgeofile,
    name = 'Mortality',
    data = UK_merg_df,
    columns = ['Name','cum_deaths_28days'],
    key_on = 'feature.properties.OBJECTID',
    fill_color = 'Greys',
    fill_opacity = 0.8,
    line_opacity = 0.2,
    legend_name = 'Mortality Rate',
    #bins = [0, 10, 15, 20, 30, 40, 50, max_death/1000]
).add_to(UKmap) '''
for region in geojson_data['features'] :
    popup = folium.Popup("Name ")
folium.LayerControl(collapsed=False).add_to(UKmap) 
UKmap


# # Dashboard

# In[20]:


from Dashboard_text import text

# creating an empty dashboard with title and icon
pn.extension(sizing_mode = "stretch_width") 
virus_pic = 'https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Ffree-photos-vectors%2Fvirus&psig=AOvVaw3xM20u-kPtVo-OhJgrX9EJ&ust=1675787071350000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCJCv5Mengf0CFQAAAAAdAAAAABAE'
vaccine_icon = " data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAACXBIWXMAAAsTAAALEwEAmpwYAAACGklEQVR4nO1ZPUsDQRB9mGirgnV+gUX+QMROERsrIWnsbG0ENTba2troT/AfCKKQM6WKIFgb4U4QbJIiIDiyZ+6KmEvuO7N78+CR4vYyO29n9mZ2AYGAI9oAKCQtGAiKSONAirXLzliKADA8AlDgFLAiON+C4SDFysmjS5NXPQiuw/Urx6UIAIkAkhSA7AEEQ9GOUQMY1RNQQhajB6gZ3BOQCIBip4CVwPmWsfV/JYAmrfww5EAEY74GJub+MEQAyJkgtCiA2hnV7tqkAGVUuOglQC392l2bPYBEAOQXARx7ABIBkP8myGkPsDLq3rQRIKvuTZvPYFYrpo0AmwDuYoS/emdD9xRoAvhJsAeodw91FWBVGZ4pz1K1cUxbF0/+5eUkqrHVRtN9dzD5FR1T4FYZVo6EdXyY1fqRN/kbHQXoKcNRVn5UJAwm39UxBUgxrvMh7vr1EADpMen/GyeAxf0+gBRt5yMRQ6yg+5zjfQDlKQCn3A8tQJQxCAa73M9bAIvrXSAFORc04ZgC+LYOrj9dclj9qQjw7PRdshfATjcFfFsigKNZBNjpfAZ9WyKA06eHTpdXBCC7UnikAPunZ974l5z85CNA69Wm+YVFb7w6ipsqKON2+J+t7Z3dcQco5gtQKpXV7zeAZRRRAPzxHExAUxDgC8ASmOA9xQ3wbYItb9weGGEdQCcl59cm2FKXKPcA5nLyTYAi4RfhgT/YjOjulwAAAABJRU5ErkJggg==" 
dashboard = pn.template.BootstrapTemplate ( title = 'Does Vaccination make less people dead in UK?', logo = vaccine_icon, header_background ='pink', sidebar_width = 150 )

# create buttons
button_home = pn.widgets.Button(name = 'Home')
button_Analyse = pn.widgets.Button(name = 'Analyse')
button_GIS = pn.widgets.Button(name = 'Map') 
button_Conclusion = pn.widgets.Button(name = 'Conclusion')
 


# add buttons to sidebar
dashboard.sidebar.append( button_home )
dashboard.sidebar.append( button_Analyse )
dashboard.sidebar.append( button_GIS )
dashboard.sidebar.append( button_Conclusion )


home_card = pn.Column( 
        pn.Card(trend_plot, title = 'The trend of Covid-19 cases during time ',active_header_background  ='gainsboro',header_color='palevioletred', sizing_mode='stretch_width'),
        pn.Card(population_case_plot, title = "Population Vs Total Cases", active_header_background  ='gainsboro',header_color='palevioletred',sizing_mode='stretch_width',collapsed =True)    
)


home = pn.Column(
        pn.Row(pn.pane.Markdown(text('home_text'))),  
        pn.Row(home_card), 
        pn.Row(pn.pane.Markdown(text('home_addition_text')),height = 50)) 

# creating a default view for the first time when we run dashboard 
main_layout = pn.Column(home) 

dashboard.main.append(main_layout)

# making different pages for different parts of analysis
vacc_mor_tab = pn.Column(
        pn.Row( pn.pane.Markdown(text('vacc_mor_text')), height = 50),
        pn.Row(vacc_mortality_plot),
        pn.Row(pn.pane.Markdown(text('vacc_mor_conc_text')), height = 50))  

region_vacc_death_tab = pn.Column(
        pn.Row( pn.pane.Markdown(text('region_vacc_text')), height = 50),
        pn.Row(gridplot(vac_death_scatter_plot, ncols = 2)))

# puting analysis pages into tabs
analyse_tab = pn.Tabs(
        ( 'Total vaccination & mortality rate',vacc_mor_tab),
        ("Regional vaccination & mortality rate",region_vacc_death_tab), dynamic=True)
 
# creating some functions to show what we want on click
def callback_home(event): 
    main_layout[0] = pn.Column(pn.Row(pn.pane.Markdown(text('home_text'))), pn.Row(trend_plot)) 
 
def callback_Analyse(event):
    main_layout[0] =  analyse_tab
       
def callback_GIS(event):
    main_layout[0] = pn.Column(pn.Row(pn.pane.Markdown(text('map_text')),height = 50),pn.Row(UKmap))  
 

def callback_conclusion(event):
    main_layout[0] = pn.Column(
        pn.Row(pn.pane.Markdown(text('conc_text')), 
        pn.Row(pn.pane.PNG(virus_pic, alt_text = 'virus image', width=500))))  
 
 

# assign callback functions to buttons
button_home.on_click(callback_home)
button_Analyse.on_click(callback_Analyse)
button_GIS.on_click(callback_GIS) 
button_Conclusion.on_click(callback_conclusion)
 
 
dashboard.show( )


# In[ ]:




