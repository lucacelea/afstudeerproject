# The Disinfection Detection System (DDS)
## Business Requirements
### Background

`To prevent the spread of corona, there’s been an increasing need in social distancing and disinfecting. There are rules set in place by the government but they are sometimes forgotten by the public. Especially disinfecting when entering stores or buildings. When you are preoccupied with daily life, it can be hard to remember.`

### Business Opportunity

`UZ Leuven have requested a system which could detect how many people entering their building actually disinfect their hands. With this data they could have more insight into the behaviours of people. With this data they will know if they have to redirect more effort (or not) into making the disinfecting of hands more likely. Such as making it more visible or other suitable options.`

###	Business Improvement Objectives

- BO-1: Measure the amount of people disinfecting their hands via a (privacy safe) camera system, to make sure this is sufficient enough.
- BO-2: When they try to increase the number of people that disinfect their hands by making changes regarding the disinfection station, the system can be used as feedback. 

### Success Metrics

- SM-1: The system can correctly detect all people in the camera frame.
- SM-2: The system can correctly detect a person using a disinfection station.
- SM-3: The data is obtained using edge computing, to ensure the visitor’s privacy and is stored securely.

### Vision Statement

`For the team responsible for corona measures it will be possible to use the DDS to obtain data regarding the use of disinfection stations. The DDS can detect people in frame and if they are likely to be disinfecting their hands. This data can be used to gauge if the disinfection stations are actually being used or not and can be used to increase the efficacy of the disinfection stations.`

### Business Risks
- RI-1:	The GDPR law is not followed correctly. (Probability = 0.1; Impact 10)
- RI-2:	The DDS could deliver inaccurate data because of unforeseeable circumstances (false positives, false negatives). (Probability = 0.85; Impact 2)

## Scope and Limitations
### Major Features
- FE-1:	Detect all people from a camera feed.
- FE-2:	Detect if detected people are using a disinfection station.
- FE-3:	Collected data is stored locally on a SD card.
- FE-4:	Collected data is passed over a secured bluetooth connection.

### Stakeholders profiles
| Stakeholder  | Major Value  |  Attitudes | Major Interests  | Constraints  |
|---|---|---|---|---|
|  Visitor | Possible increased safety  | Privacy concerns  | Increased safety  | None identified  |
| Corona responsible employee/team  |  More information regarding their visitor’s behaviour | Gauge the efficacy of disinfection station  | Increased safety; Maintain visitor’s privacy  | Need information regarding the usage of the DDS  |
| Management committee  | More information regarding their visitor’s behaviour  |  Pricing concerns; Gauge the efficacy of disinfection station |  Increased safety; Maintain visitor’s privacy; Lowest possible cost | Assign responsible employee/team for operating the DDS  |
| Siegmund Leducq, Johan Strypsteen, David Vandenbroeck  |  Client |  Efficiency of the system |  Working system | None identified  |

# Software requirements specifications
## Users and Characteristics

- Visitor: a person who visits the hospital. This person will be detected if the person enters the entrance hall and if the person uses the disinfection station.

- Corona responsible employee/team: employee of the hospital. This team will operate the DDS and analyse the data.

- Management committee: the committee responsible for making employees available to use the DDS and determine the funds for project.

## Operating Environment Constraints

- OE-1:	Camera has to be as vertical and high as possible
- OE-2: The system has to run on the delivered hardware (Nvidia Xavier, Nvidea Nano or Google Coral)

## Design and Implementation Constraints

- CO-1:	The system will be designed in Python.
- CO-2:	The system shall use OpenCV as a library.
- CO-3:	The system shall use Tensorflow as a framework.

## Assumptions 

- Disinfection station is visible and operational.
- The camera system is operational and of decent enough quality to be able to detect persons.

# External Interface Requirements
## User Interfaces

- UI-1:	TKinter	

##	 Software Interfaces

- SI-1:	Linux

## Communications Interfaces

- CI-1: The DDS stores data on a local SD card.
- CI-2: The DDS can transfer data via a secure bluetooth connection.
