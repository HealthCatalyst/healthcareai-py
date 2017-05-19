# Profiling Your Data

## Background - Why Profile?

### TL;DR

Often models are trained on retrospective data, which is typically highly available and clean. When models are deployed,  realtime production data is never as clean or available. The **Feature Availabilty Profiler** can expose some of these problems.

### More Details

Let's say that we developed and trained a predictive machine learning model on retrospective data from an inpatient unit. We may be trying to identfy patients at risk for a certain outcome during their stay in the hospital.

Let's presume that our model is performing well with an [ROC AUC](https://healthcare.ai/model-evaluation-using-roc-curves/) of 0.83.

Things appear to be going smoothly, so we push our model out to production to make predictions on current patient data. We check back in on our model and find that the *in the wild* ROC AUC has fallen to 0.67.

What's happening here? It is likely we might be experiencing some [data leakage](https://healthcare.ai/data-leakage-in-healthcare-machine-learning/). This means that in our retrospective training data we might have had a table like this:

| AdmitDateTime       | MRN  | Height | Weight | Age  | Gender | LabRBC | LabHematocrit |
| ------------------- | ---- | ------ | ------ | ---- | ------ | ------ | ------------- |
| 2017-04-01 00:05:44 | 3    | 156    | 68     | 56   | F      | 5.3    | 47            |
| 2017-04-01 00:06:33 | 4    | 166    | 94     | 33   | M      | 6.1    | 39            |
| 2017-04-02 00:07:59 | 5    | 134    | 47     | 88   | M      | 5.9    | 55            |
| 2017-04-02 00:13:07 | 1    | 180    | 66     | 56   | F      | 5.5    | 41            |
| 2017-04-03 00:21:12 | 2    | 177    | 57     | 45   | M      | 6.3    | 48            |

Let's imagine that our model used height, weight, age, gender, red blood count and hematocrit as features.

However, realtime data is never this clean and available. If we were to look at records of patients who are currently in hospital, we might see something like this:

| AdmitDateTime       | MRN  | Height | Weight | Age  | Gender | LabRBC | LabHematocrit |
| ------------------- | ---- | ------ | ------ | ---- | ------ | ------ | ------------- |
| 2017-04-01 00:05:44 | 1    | 156    | 68     | 56   | F      | 5.3    | 47            |
| 2017-04-01 00:06:33 | 2    | 166    | 94     | 33   | M      | 6.1    | 39            |
| 2017-04-02 00:07:59 | 3    | 134    | 47     | 88   | M      |        |               |
| 2017-04-02 00:13:07 | 4    | 180    | 66     | 56   | F      |        |               |
| 2017-04-03 00:21:12 | 5    | 177    | 57     |      | M      |        |               |

Here we see a few patients (id 1 & 2) have been inpatients for a few days and have had CBC labs done. However if we look at patients 3-5 we see that they have not been admits as long and are missing some labs. Patient 5 was admitted a few hours ago and we don't even have access to his age yet.

When we run the predictive model on these patients with missing values the model has less information about each patient and will therefore make a less accurate prediction.

## A solution

As we help users with healthcare.ai we keep seeing this problem. So, we built a tool to help uncover this problem. We call it the **Feature Availability Profiler**.

## Using the Feature Availability Profiler

The availability profiler assumes that your data has two date/time columns. One is the patient's admit time. The other is the timestamp when the data was last saved to the database. Both date fields are needed for caluculating how long a patient has been in the unit. If you do not have the second date, you could add a column with the current time to the dataframe.

1. Load your data into a datarame.
2. Pass your dataframe into the profiler as such:
```
feature_availability_profiler(dataframe, admit_col_name='AdmitDTS', last_load_col_name='LastLoadDTS')
```

You will then see a graph like the one below that shows you each feature (aka database field) and it's percentage of availability as time goes on.

# TODO: add graph and remove this heading

![Sample output from Feature Availabilty Profiler](foo.png)