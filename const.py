
# 'algorithm', 'conference', 'country', 'field', 'location', 'metrics', 'misc', 'organisation', 'person', 'product', 'programlang', 'researcher', 'task', 'university'
AI_CLASSS = {
    'algorithm':'algorithm', 
    'conference': 'conference', 
    'country': 'country', 
    'field': 'field', 
    'location': 'location', 
    'metrics': 'metrics', 
    'misc': 'miscellaneous', 
    'organisation': 'organization', 
    'person': 'person', 
    'product': 'product', 
    'programlang': 'programming language', 
    'researcher': 'researcher', 
    'task': 'task', 
    'university': 'university'
}

AI_PROMPTS = [
    """You are a mention detection assistant. Your task is to identify named entities in a given text and wrap each entity mention with <entity> and </entity> tags. 

    Instructions:
    1. Return the *exact same text*, but with each named entity enclosed in <entity>...</entity> tags.
    2. Named entities are from AI domain, including 'algorithm', 'conference', 'country', 'field', 'location', 'metrics', 'miscellaneous', 'organization', 'person', 'product', 'programming language', 'researcher', 'task', 'university'.
    3. If there are multiple words in an entity (e.g., “Deep Neural Network”), wrap the entire phrase: <entity>Deep Neural Network</entity>.
    4. Keep the rest of the text unchanged.
    5. Do not add any extra commentary or formatting. Only return the modified text.
    6. If there are no entities, simply return the original text without any tags.""",
    
    """You are an entity typing assistant. You will receive text where certain substrings are wrapped in <entity>...</entity> tags. These substrings are entity mentions. 

    Your task:
    1. For each mention wrapped in <entity>...</entity>, determine the most suitable entity type from the following list:
    - algorithm
    - conference
    - country
    - field
    - location
    - metrics
    - miscellaneous
    - organization
    - person
    - product
    - programming language
    - researcher
    - task
    - university
    - other  (use this if the mention does not fit any of the above)

    2. Convert each <entity>...</entity> tag into <entity type="TYPE">...</entity>, where TYPE is one of the listed options. 
    - Example: If the mention is "Deep Neural Network" and you decide it’s an 'algorithm', change <entity>Deep Neural Network</entity> to <entity type="algorithm">Deep Neural Network</entity>.

    3. Leave all other text (outside the entity tags) exactly the same.

    4. If a mention is ambiguous or does not clearly fit any of the predefined types, classify it as "other".

    5. Output only the modified text, with the <entity> tags updated to include the type. Do not add any extra commentary, code fences, or explanation."""
]


# 'academicjournal', 'astronomicalobject', 'award', 'chemicalcompound', 'chemicalelement', 'country', 'discipline', 'enzyme', 'event', 'location', 'misc', 'organisation', 'person', 'protein', 'scientist', 'theory', 'university'
SCIENCE_CLASS = {
    'academicjournal': 'academic journal', 
    'astronomicalobject': 'astronomical object', 
    'award': 'award', 
    'chemicalcompound': 'chemical compound', 
    'chemicalelement': 'chemical element', 
    'country': 'country', 
    'discipline': 'discipline', 
    'enzyme': 'enzyme', 
    'event': 'event', 
    'location': 'location', 
    'misc': 'miscellaneous', 
    'organisation': 'organization', 
    'person': 'person', 
    'protein': 'protein', 
    'scientist': 'scientist', 
    'theory': 'theory', 
    'university': 'university'
}

SCIENCE_PROMPTS = [
    """You are a mention detection assistant. Your task is to identify named entities in a given text and wrap each entity mention with <entity> and </entity> tags. 

    Instructions:
    1. Return the *exact same text*, but with each named entity enclosed in <entity>...</entity> tags.
    2. Named entities are from Science domain, including 'academic journal', 'astronomical object', 'award', 'chemical compound', 'chemical element', 'country', 'discipline', 'enzyme', 'event', 'location', 'miscellaneous', 'organization', 'person', 'protein', 'scientist', 'theory', 'university'.
    3. If there are multiple words in an entity (e.g., “Temple University”), wrap the entire phrase: <entity>Temple University</entity>.
    4. Keep the rest of the text unchanged.
    5. Do not add any extra commentary or formatting. Only return the modified text.
    6. If there are no entities, simply return the original text without any tags.""",
    
    """You are an entity typing assistant. You will receive text where certain substrings are wrapped in <entity>...</entity> tags. These substrings are entity mentions. 

    Your task:
    1. For each mention wrapped in <entity>...</entity>, determine the most suitable entity type from the following list:
    - academic journal
    - astronomical object
    - award
    - chemical compound
    - chemical element
    - country
    - discipline
    - enzyme
    - event
    - location
    - miscellaneous
    - organization
    - person
    - protein
    - scientist
    - theory
    - university
    - other  (use this if the mention does not fit any of the above)

    2. Convert each <entity>...</entity> tag into <entity type="TYPE">...</entity>, where TYPE is one of the listed options. 
    - Example: If the mention is "Temple University" and you decide it’s an 'organization', change <entity>Temple University</entity> to <entity type="organization">Temple University</entity>.

    3. Leave all other text (outside the entity tags) exactly the same.

    4. If a mention is ambiguous or does not clearly fit any of the predefined types, classify it as "other".

    5. Output only the modified text, with the <entity> tags updated to include the type. Do not add any extra commentary, code fences, or explanation."""
]

# 'award', 'book', 'country', 'event', 'literarygenre', 'location', 'magazine', 'misc', 'organisation', 'person', 'poem', 'writer'
LITERATURE_CLASSS = {
    'award': 'award', 
    'book': 'book', 
    'country': 'country', 
    'event': 'event', 
    'literarygenre': 'literary genre', 
    'location': 'location', 
    'magazine': 'magazine', 
    'misc': 'miscellaneous', 
    'organisation': 'organization',
    'person': 'person', 
    'poem': 'poem', 
    'writer': 'writer'
}

# 'award', 'book', 'country', 'event', 'literary genre', 'location', 'magazine', 'miscellaneous', 'organization', 'person', 'poem', 'writer'
LITERATURE_PROMPTS = [
    """You are a mention detection assistant. Your task is to identify named entities in a given text and wrap each entity mention with <entity> and </entity> tags. 

    Instructions:
    1. Return the *exact same text*, but with each named entity enclosed in <entity>...</entity> tags.
    2. Named entities are from Literature domain, including 'award', 'book', 'country', 'event', 'literary genre', 'location', 'magazine', 'miscellaneous', 'organization', 'person', 'poem', 'writer'.
    3. If there are multiple words in an entity (e.g., “New York”), wrap the entire phrase: <entity>New York</entity>.
    4. Keep the rest of the text unchanged.
    5. Do not add any extra commentary or formatting. Only return the modified text.
    6. If there are no entities, simply return the original text without any tags.""",
    
    """You are an entity typing assistant. You will receive text where certain substrings are wrapped in <entity>...</entity> tags. These substrings are entity mentions. 

    Your task:
    1. For each mention wrapped in <entity>...</entity>, determine the most suitable entity type from the following list:
    - award
    - book
    - country
    - event
    - literary genre
    - location
    - magazine
    - miscellaneous
    - organization
    - person
    - poem
    - writer
    - other  (use this if the mention does not fit any of the above)

    2. Convert each <entity>...</entity> tag into <entity type="TYPE">...</entity>, where TYPE is one of the listed options. 
    - Example: If the mention is "New York" and you decide it’s a 'location', change <entity>New York</entity> to <entity type="location">New York</entity>.

    3. Leave all other text (outside the entity tags) exactly the same.

    4. If a mention is ambiguous or does not clearly fit any of the predefined types, classify it as "other".

    5. Output only the modified text, with the <entity> tags updated to include the type. Do not add any extra commentary, code fences, or explanation."""
]


# 'album', 'award', 'band', 'country', 'event', 'location', 'misc', 'musicalartist', 'musicalinstrument', 'musicgenre', 'organisation', 'person', 'song'
MUSIC_CLASSS = {
    'album': 'album', 
    'award': 'award', 
    'band': 'band', 
    'country': 'country', 
    'event': 'event', 
    'location': 'location', 
    'misc': 'miscellaneous', 
    'musicalartist': 'musical artist', 
    'musicalinstrument': 'musical instrument', 
    'musicgenre': 'music genre', 
    'organisation': 'organization', 
    'person': 'person', 
    'song': 'song'
}

# 'album', 'award', 'band', 'country', 'event', 'location', 'miscellaneous', 'musical artist', 'musical instrument', 'music genre', 'organization', 'person', 'song'
MUSIC_PROMPTS = [
    """You are a mention detection assistant. Your task is to identify named entities in a given text and wrap each entity mention with <entity> and </entity> tags. 

    Instructions:
    1. Return the *exact same text*, but with each named entity enclosed in <entity>...</entity> tags.
    2. Named entities are from Music domain, including 'album', 'award', 'band', 'country', 'event', 'location', 'miscellaneous', 'musical artist', 'musical instrument', 'music genre', 'organization', 'person', 'song'.
    3. If there are multiple words in an entity (e.g., “Grammy Award”), wrap the entire phrase: <entity>Grammy Award</entity>.
    4. Keep the rest of the text unchanged.
    5. Do not add any extra commentary or formatting. Only return the modified text.
    6. If there are no entities, simply return the original text without any tags.""",
    
    """You are an entity typing assistant. You will receive text where certain substrings are wrapped in <entity>...</entity> tags. These substrings are entity mentions. 

    Your task:
    1. For each mention wrapped in <entity>...</entity>, determine the most suitable entity type from the following list:
    - album
    - award
    - band
    - country
    - event
    - location
    - miscellaneous
    - musical artist
    - musical instrument
    - music genre
    - organization
    - person
    - song
    - other  (use this if the mention does not fit any of the above)

    2. Convert each <entity>...</entity> tag into <entity type="TYPE">...</entity>, where TYPE is one of the listed options. 
    - Example: If the mention is "Grammy Award" and you decide it’s an 'award', change <entity>Grammy Award</entity> to <entity type="award">Grammy Award</entity>.

    3. Leave all other text (outside the entity tags) exactly the same.

    4. If a mention is ambiguous or does not clearly fit any of the predefined types, classify it as "other".

    5. Output only the modified text, with the <entity> tags updated to include the type. Do not add any extra commentary, code fences, or explanation."""
]

# 'country', 'election', 'event', 'location', 'misc', 'organisation', 'person', 'politicalparty', 'politician'
POLITICS_CLASSS = {
    'country': 'country', 
    'election': 'election', 
    'event': 'event', 
    'location': 'location', 
    'misc': 'miscellaneous', 
    'organisation': 'organization',
    'person': 'person',
    'politicalparty': 'political party',
    'politician': 'politician'
}

# 'country', 'election', 'event', 'location', 'miscellaneous', 'organization', 'person', 'political party', 'politician'
POLITICS_PROMPTS = [
    """You are a mention detection assistant. Your task is to identify named entities in a given text and wrap each entity mention with <entity> and </entity> tags. 

    Instructions:
    1. Return the *exact same text*, but with each named entity enclosed in <entity>...</entity> tags.
    2. Named entities are from Politics domain, including 'country', 'election', 'event', 'location', 'miscellaneous', 'organization', 'person', 'political party', 'politician'.
    3. If there are multiple words in an entity (e.g., “Republican Party”), wrap the entire phrase: <entity>Republican Party</entity>.
    4. Keep the rest of the text unchanged.
    5. Do not add any extra commentary or formatting. Only return the modified text.
    6. If there are no entities, simply return the original text without any tags.""",
    
    """You are an entity typing assistant. You will receive text where certain substrings are wrapped in <entity>...</entity> tags. These substrings are entity mentions. 

    Your task:
    1. For each mention wrapped in <entity>...</entity>, determine the most suitable entity type from the following list:
    - country
    - election
    - event
    - location
    - miscellaneous
    - organization
    - person
    - political party
    - politician
    - other  (use this if the mention does not fit any of the above)

    2. Convert each <entity>...</entity> tag into <entity type="TYPE">...</entity>, where TYPE is one of the listed options. 
    - Example: If the mention is "Republican Party" and you decide it’s a 'political party', change <entity>Republican Party</entity> to <entity type="political party">Republican Party</entity>.

    3. Leave all other text (outside the entity tags) exactly the same.

    4. If a mention is ambiguous or does not clearly fit any of the predefined types, classify it as "other".

    5. Output only the modified text, with the <entity> tags updated to include the type. Do not add any extra commentary, code fences, or explanation."""
]

CoT_CLIMATE_CHAIN_PROMPTS_ZERO = [
    """You are a mention detection assistant. Your task is to identify named entities in a given text and wrap each entity mention with <entity> and </entity> tags. 

    Instructions:
    1. Return the *exact same text*, but with each named entity enclosed in <entity>...</entity> tags.
    2. Named entities are from Climate domain. The included entity types and their definitions are: 
    - 'project': A project refers to the scientific program, field campaign, or project from which the data were collected.
    - 'location': A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
    - 'model': A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
    - 'experiment': An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
    - 'platform': A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
    - 'instrument': A instrument is a device used to measure, observe, or calculate.
    - 'provider': A provider is an organization, an academic institution or a commercial company.
    - 'variable': A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
    - 'weather event': A weather event refers to a specific atmospheric phenomenon or condition, such as storms, hurricanes, droughts, or heatwaves, occurring over a short period of time and often having measurable impacts on the environment or society.
    - 'natural hazard': A natural hazard is a potentially damaging physical event, such as earthquakes, volcanic eruptions, floods, or landslides, that arises from natural processes and may pose risks to human life, property, or the environment.
    - 'teleconnection': A teleconnection is a statistical relationship or linkage between climate anomalies in different geographic regions, often driven by large-scale atmospheric or oceanic patterns, such as El Niño or the North Atlantic Oscillation.
    - 'ocean circulation': Ocean circulation refers to the large-scale movement of water masses within the oceans, driven by factors such as wind, salinity, and temperature gradients, and playing a critical role in regulating Earth’s climate system.
    3. If there are multiple words in an entity (e.g., "the temperature is"), wrap the entire phrase: "the <entity>temperature</entity> is".
    4. Keep the rest of the text unchanged.
    5. Do not add any extra commentary or formatting. Only return the modified text.
    6. If there are no entities, simply return the original text without any tags.
    
    Think step by step and explain why a span is tagged as a mention.""",
    
    """You are an entity typing assistant. You will receive text where certain substrings are wrapped in <entity>...</entity> tags. These substrings are entity mentions. 

    Your task:
    1. For each mention wrapped in <entity>...</entity>, determine the most suitable entity type. The interested types and their definitions are:
    - 'project': A project refers to the scientific program, field campaign, or project from which the data were collected.
    - 'location': A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
    - 'model': A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
    - 'experiment': An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
    - 'platform': A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
    - 'instrument': A instrument is a device used to measure, observe, or calculate.
    - 'provider': A provider is an organization, an academic institution or a commercial company.
    - 'variable': A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
    - 'weather event': A weather event refers to a specific atmospheric phenomenon or condition, such as storms, hurricanes, droughts, or heatwaves, occurring over a short period of time and often having measurable impacts on the environment or society.
    - 'natural hazard': A natural hazard is a potentially damaging physical event, such as earthquakes, volcanic eruptions, floods, or landslides, that arises from natural processes and may pose risks to human life, property, or the environment.
    - 'teleconnection': A teleconnection is a statistical relationship or linkage between climate anomalies in different geographic regions, often driven by large-scale atmospheric or oceanic patterns, such as El Niño or the North Atlantic Oscillation.
    - 'ocean circulation': Ocean circulation refers to the large-scale movement of water masses within the oceans, driven by factors such as wind, salinity, and temperature gradients, and playing a critical role in regulating Earth’s climate system.
    - other  (use this if the mention does not fit any of the above)

    2. Convert each <entity>...</entity> tag into <entity type="TYPE">...</entity>, where TYPE is one of the listed options. 
    - Example: If the mention is "temperature" and you decide it’s an 'variable', change <entity>temperature</entity> to <entity type="variable">temperature</entity>.

    3. Leave all other text (outside the entity tags) exactly the same.

    4. If a mention is ambiguous or does not clearly fit any of the predefined types, classify it as "other".

    5. Output only the modified text, with the <entity> tags updated to include the type. Do not add any extra commentary, code fences, or explanation.
    
    Think step by step and explain why this mention is classified as a specific entity type."""
]





CLIMATE_CHAIN_PROMPTS_ZERO = [
    """You are a mention detection assistant. Your task is to identify named entities in a given text and wrap each entity mention with <entity> and </entity> tags. 

    Instructions:
    1. Return the *exact same text*, but with each named entity enclosed in <entity>...</entity> tags.
    2. Named entities are from Climate domain. The included entity types and their definitions are: 
    - 'project': A project refers to the scientific program, field campaign, or project from which the data were collected.
    - 'location': A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
    - 'model': A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
    - 'experiment': An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
    - 'platform': A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
    - 'instrument': A instrument is a device used to measure, observe, or calculate.
    - 'provider': A provider is an organization, an academic institution or a commercial company.
    - 'variable': A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
    - 'weather event': A weather event refers to a specific atmospheric phenomenon or condition, such as storms, hurricanes, droughts, or heatwaves, occurring over a short period of time and often having measurable impacts on the environment or society.
    - 'natural hazard': A natural hazard is a potentially damaging physical event, such as earthquakes, volcanic eruptions, floods, or landslides, that arises from natural processes and may pose risks to human life, property, or the environment.
    - 'teleconnection': A teleconnection is a statistical relationship or linkage between climate anomalies in different geographic regions, often driven by large-scale atmospheric or oceanic patterns, such as El Niño or the North Atlantic Oscillation.
    - 'ocean circulation': Ocean circulation refers to the large-scale movement of water masses within the oceans, driven by factors such as wind, salinity, and temperature gradients, and playing a critical role in regulating Earth’s climate system.
    3. If there are multiple words in an entity (e.g., "the temperature is"), wrap the entire phrase: "the <entity>temperature</entity> is".
    4. Keep the rest of the text unchanged.
    5. Do not add any extra commentary or formatting. Only return the modified text.
    6. If there are no entities, simply return the original text without any tags.""",
    
    """You are an entity typing assistant. You will receive text where certain substrings are wrapped in <entity>...</entity> tags. These substrings are entity mentions. 

    Your task:
    1. For each mention wrapped in <entity>...</entity>, determine the most suitable entity type. The interested types and their definitions are:
    - 'project': A project refers to the scientific program, field campaign, or project from which the data were collected.
    - 'location': A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
    - 'model': A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
    - 'experiment': An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
    - 'platform': A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
    - 'instrument': A instrument is a device used to measure, observe, or calculate.
    - 'provider': A provider is an organization, an academic institution or a commercial company.
    - 'variable': A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
    - 'weather event': A weather event refers to a specific atmospheric phenomenon or condition, such as storms, hurricanes, droughts, or heatwaves, occurring over a short period of time and often having measurable impacts on the environment or society.
    - 'natural hazard': A natural hazard is a potentially damaging physical event, such as earthquakes, volcanic eruptions, floods, or landslides, that arises from natural processes and may pose risks to human life, property, or the environment.
    - 'teleconnection': A teleconnection is a statistical relationship or linkage between climate anomalies in different geographic regions, often driven by large-scale atmospheric or oceanic patterns, such as El Niño or the North Atlantic Oscillation.
    - 'ocean circulation': Ocean circulation refers to the large-scale movement of water masses within the oceans, driven by factors such as wind, salinity, and temperature gradients, and playing a critical role in regulating Earth’s climate system.
    - other  (use this if the mention does not fit any of the above)

    2. Convert each <entity>...</entity> tag into <entity type="TYPE">...</entity>, where TYPE is one of the listed options. 
    - Example: If the mention is "temperature" and you decide it’s an 'variable', change <entity>temperature</entity> to <entity type="variable">temperature</entity>.

    3. Leave all other text (outside the entity tags) exactly the same.

    4. If a mention is ambiguous or does not clearly fit any of the predefined types, classify it as "other".

    5. Output only the modified text, with the <entity> tags updated to include the type. Do not add any extra commentary, code fences, or explanation."""
]



CLIMATE_CHAIN_PROMPTS_FEW = [
    """### Task
    You are a mention detection assistant. Your task is to identify named entities in a given text and wrap each entity mention with <entity> and </entity> tags. 

    Instructions:
    1. Return the *exact same text*, but with each named entity enclosed in <entity>...</entity> tags.
    2. Named entities are from Climate domain. The included entity types and their definitions are: 
    - 'project': A project refers to the scientific program, field campaign, or project from which the data were collected.
    - 'location': A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
    - 'model': A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
    - 'experiment': An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
    - 'platform': A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
    - 'instrument': A instrument is a device used to measure, observe, or calculate.
    - 'provider': A provider is an organization, an academic institution or a commercial company.
    - 'variable': A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
    - 'weather event': A weather event refers to a specific atmospheric phenomenon or condition, such as storms, hurricanes, droughts, or heatwaves, occurring over a short period of time and often having measurable impacts on the environment or society.
    - 'natural hazard': A natural hazard is a potentially damaging physical event, such as earthquakes, volcanic eruptions, floods, or landslides, that arises from natural processes and may pose risks to human life, property, or the environment.
    - 'teleconnection': A teleconnection is a statistical relationship or linkage between climate anomalies in different geographic regions, often driven by large-scale atmospheric or oceanic patterns, such as El Niño or the North Atlantic Oscillation.
    - 'ocean circulation': Ocean circulation refers to the large-scale movement of water masses within the oceans, driven by factors such as wind, salinity, and temperature gradients, and playing a critical role in regulating Earth’s climate system.
    3. If there are multiple words in an entity (e.g., "the temperature is"), wrap the entire phrase: "the <entity>temperature</entity> is".
    4. Keep the rest of the text unchanged.
    5. Do not add any extra commentary or formatting. Only return the modified text.
    6. If there are no entities, simply return the original text without any tags.
    
    ### Examples
    Input: The CMIP6 experiments were run by the UK Met Office and included the simulation of front patterns in the North Atlantic.
    Output: The <entity>CMIP6</entity> experiments were run by the <entity>UK Met Office</entity> and included the simulation of front patterns in the <entity>North Atlantic</entity>.
    Input: The MODIS on the Terra satellite has been widely used to measure surface temperatures over desert regions.
    Output: The <entity>MODIS</entity> on the <entity>Terra</entity> satellite has been widely used to measure <entity>surface temperatures</entity> over <entity>desert regions</entity>.
    Input: Hurricane Katrina was correctly captured by GFDL climate models provided by NOAA.
    Output: <entity>Hurricane Katrina</entity> was correctly captured by <entity>GFDL climate models</entity> provided by <entity>NOAA</entity>.
    Input: The study revealed relationship between El Niño in the Pacific and decadal droughts in the Sahel region of Africa.
    Output: The study revealed relationship between <entity>El Niño</entity> in the Pacific and decadal <entity>droughts</entity> in the <entity>Sahel</entity> region of <entity>Africa</entity>.
    Input: The Earth System Model outputs datasets on carbon flux, which are crucial for understanding global climate dynamics.
    Output: The <entity>Earth System Model</entity> outputs datasets on <entity>carbon flux</entity>, which are crucial for understanding global climate dynamics.
    Input: Data from the PIRATA project were used to validate the CESM2 and WRF models simulating ocean-atmosphere interactions in the tropical Atlantic. 
    Output: Data from the <entity>PIRATA</entity> project were used to validate the <entity>CESM2</entity> and <entity>WRF</entity> models simulating ocean-atmosphere interactions in the <entity>tropical Atlantic</entity>.
    Input: The HadCRUT dataset, provided by the Climatic Research Unit, includes temperature measurements at various locations globally.
    Output: The <entity>HadCRUT</entity> dataset, provided by the <entity>Climatic Research Unit</entity>, includes <entity>temperature measurements</entity> at various locations globally.
    Input: The Scripps Institution of Oceanography mounted Argo floats to measure salinity and temperature in the Southern Ocean.
    Output: The <entity>Scripps Institution of Oceanography</entity> mounted <entity>Argo floats</entity> to measure <entity>salinity</entity> and <entity>temperature</entity> in the <entity>Southern Ocean</entity>.
    Input: ECMWF model output was used in the European Climate Assessment project to analyse tornadoes events in the Alps and its impact on floods.
    Output: <entity>ECMWF</entity> model output was used in the <entity>European Climate Assessment</entity> project to analyse <entity>tornadoes</entity> events in the <entity>Alps</entity> and its impact on <entity>floods</entity>.
    Input: The ensemble mean bias in the position of the jet in a subset of nine available atmosphere-only AMIP simulations was found to be 28% smaller than the historical CMIP5 simulations with corresponding atmospheric models.
    Output: The ensemble mean bias in the position of the jet in a subset of nine available <entity>atmosphere-only</entity> <entity>AMIP</entity> simulations was found to be 28% smaller than the historical <entity>CMIP5</entity> simulations with corresponding <entity>atmospheric models</entity>.
    """,
    
    """### Task
    You are an entity typing assistant. You will receive text where certain substrings are wrapped in <entity>...</entity> tags. These substrings are entity mentions. 

    Your task:
    1. For each mention wrapped in <entity>...</entity>, determine the most suitable entity type. The interested types and their definitions are:
    - 'project': A project refers to the scientific program, field campaign, or project from which the data were collected.
    - 'location': A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
    - 'model': A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
    - 'experiment': An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
    - 'platform': A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
    - 'instrument': A instrument is a device used to measure, observe, or calculate.
    - 'provider': A provider is an organization, an academic institution or a commercial company.
    - 'variable': A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
    - 'weather event': A weather event refers to a specific atmospheric phenomenon or condition, such as storms, hurricanes, droughts, or heatwaves, occurring over a short period of time and often having measurable impacts on the environment or society.
    - 'natural hazard': A natural hazard is a potentially damaging physical event, such as earthquakes, volcanic eruptions, floods, or landslides, that arises from natural processes and may pose risks to human life, property, or the environment.
    - 'teleconnection': A teleconnection is a statistical relationship or linkage between climate anomalies in different geographic regions, often driven by large-scale atmospheric or oceanic patterns, such as El Niño or the North Atlantic Oscillation.
    - 'ocean circulation': Ocean circulation refers to the large-scale movement of water masses within the oceans, driven by factors such as wind, salinity, and temperature gradients, and playing a critical role in regulating Earth’s climate system.
    - other  (use this if the mention does not fit any of the above)

    2. Convert each <entity>...</entity> tag into <entity type="TYPE">...</entity>, where TYPE is one of the listed options. 
    - Example: If the mention is "temperature" and you decide it’s an 'variable', change <entity>temperature</entity> to <entity type="variable">temperature</entity>.

    3. Leave all other text (outside the entity tags) exactly the same.

    4. If a mention is ambiguous or does not clearly fit any of the predefined types, classify it as "other".

    5. Output only the modified text, with the <entity> tags updated to include the type. Do not add any extra commentary, code fences, or explanation.
    
    ### Examples
    Input: The <entity>CMIP6</entity> experiments were run by the <entity>UK Met Office</entity> and included the simulation of front patterns in the <entity>North Atlantic</entity>.
    Output: The <entity type="project">CMIP6</entity> experiments were run by the <entity type="provider">UK Met Office</entity> and included the simulation of front patterns in the <entity type="location">North Atlantic</entity>.
    Input: The <entity>MODIS</entity> on the <entity>Terra</entity> satellite has been widely used to measure <entity>surface temperatures</entity> over <entity>desert regions</entity>.
    Output: The <entity type="instrument">MODIS</entity> on the <entity type="platform">Terra</entity> satellite has been widely used to measure <entity type="variable">surface temperatures</entity> over <entity type="location">desert regions</entity>.
    Input: <entity>Hurricane Katrina</entity> was correctly captured by <entity>GFDL climate models</entity> provided by <entity>NOAA</entity>.
    Output: <entity type="weather event">Hurricane Katrina</entity> was correctly captured by <entity type="model">GFDL climate models</entity> provided by <entity type="provider">NOAA</entity>.
    Input: The study revealed relationship between <entity>El Niño</entity> in the Pacific and <entity>decadal</entity> droughts in the <entity>Sahel</entity> region of <entity>Africa</entity>.
    Output: The study revealed relationship between <entity type="teleconnection">El Niño</entity> in the Pacific and decadal <entity type="natural hazard">droughts</entity> in the <entity type="location">Sahel</entity> region of <entity type="location">Africa</entity>.
    Input: The <entity>Earth System Model</entity> outputs datasets on <entity>carbon flux</entity>, which are crucial for understanding global climate dynamics.
    Output: The <entity type="model">Earth System Model</entity> outputs datasets on <entity type="variable">carbon flux</entity>, which are crucial for understanding global climate dynamics.
    Input: Data from the <entity>PIRATA</entity> project were used to validate the <entity>CESM2</entity> and <entity>WRF</entity> models simulating ocean-atmosphere interactions in the <entity>tropical Atlantic</entity>.
    Output: Data from the <entity type="project">PIRATA</entity> project were used to validate the <entity type="model">CESM2</entity> and <entity type="model">WRF</entity> models simulating ocean-atmosphere interactions in the <entity type="location">tropical Atlantic</entity>.
    Input: The <entity>HadCRUT</entity> dataset, provided by the <entity>Climatic Research Unit</entity>, includes <entity>temperature measurements</entity> at various locations globally.
    Output: The <entity type="project">HadCRUT</entity> dataset, provided by the <entity type="provider">Climatic Research Unit</entity>, includes <entity type="variable">temperature measurements</entity> at various locations globally.
    Input: The <entity>Scripps Institution of Oceanography</entity> mounted <entity>Argo floats</entity> to measure <entity>salinity</entity> and <entity>temperature</entity> in the <entity>Southern Ocean</entity>.
    Output: The <entity type="provider">Scripps Institution of Oceanography</entity> mounted <entity type="instrument">Argo floats</entity> to measure <entity type="variable">salinity</entity> and <entity type="variable">temperature</entity> in the <entity type="location">Southern Ocean</entity>.
    Input: <entity>ECMWF</entity> model output was used in the <entity>European Climate Assessment</entity> project to analyse <entity>tornadoes</entity> events in the <entity>Alps</entity> and its impact on <entity>floods</entity>.
    Output: <entity type="model">ECMWF</entity> model output was used in the <entity type="project">European Climate Assessment</entity> project to analyse <entity type="weather event">tornadoes</entity> events in the <entity type="location">Alps</entity> and its impact on <entity type="natural hazard">floods</entity>.
    Input: The ensemble mean bias in the position of the jet in a subset of nine available <entity>atmosphere-only</entity> <entity>AMIP</entity> simulations was found to be 28% smaller than the historical <entity>CMIP5</entity> simulations with corresponding <entity>atmospheric models</entity>.
    Output: The ensemble mean bias in the position of the jet in a subset of nine available <entity type="experiment">atmosphere-only</entity> <entity type="experiment">AMIP</entity> simulations was found to be 28% smaller than the historical <entity type="project">CMIP5</entity> simulations with corresponding <entity type="model">atmospheric models</entity>.
    """
]

ENTITY_TYPING_PROMPT_ZERO = """Classify all entity mentions in brackets into the corresponding types based on the context. The interested types and their definitions are:
    - 'project': A project refers to the scientific program, field campaign, or project from which the data were collected.
    - 'location': A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
    - 'model': A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
    - 'experiment': An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
    - 'platform': A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
    - 'instrument': A instrument is a device used to measure, observe, or calculate.
    - 'provider': A provider is an organization, an academic institution or a commercial company.
    - 'variable': A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
    - 'weather event': A weather event refers to a specific atmospheric phenomenon or condition, such as storms, hurricanes, droughts, or heatwaves, occurring over a short period of time and often having measurable impacts on the environment or society.
    - 'natural hazard': A natural hazard is a potentially damaging physical event, such as earthquakes, volcanic eruptions, floods, or landslides, that arises from natural processes and may pose risks to human life, property, or the environment.
    - 'teleconnection': A teleconnection is a statistical relationship or linkage between climate anomalies in different geographic regions, often driven by large-scale atmospheric or oceanic patterns, such as El Niño or the North Atlantic Oscillation.
    - 'ocean circulation': Ocean circulation refers to the large-scale movement of water masses within the oceans, driven by factors such as wind, salinity, and temperature gradients, and playing a critical role in regulating Earth’s climate system.
    - other  (use this if the mention does not fit any of the above)"""