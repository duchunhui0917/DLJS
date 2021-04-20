import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./sort_jobs.csv')
df['total_duration'] = df['duration'] * df['instances_num']

df1 = pd.DataFrame(columns=['tasks_num', 'instances_num', 'submit_time', 'duration'])
df1['tasks_num'] = df.groupby(['submit_time', 'job_id']).size()
df1['instances_num'] = df.groupby(['submit_time', 'job_id'])['instances_num'].sum()
df1['submit_time'] = df.groupby(['submit_time', 'job_id'])['submit_time'].mean()
df1['duration'] = df.groupby(['submit_time', 'job_id'])['total_duration'].sum()

n = 10
df2 = pd.DataFrame(columns=['tasks_num', 'instances_num', 'arrival_time_difference', 'duration'])
print(len(df1))
for i in range(0, len(df1), n):
    df2['tasks_num'].loc[i] = df1.iloc[i:i+n, 0].sum()
    df2['instances_num'].loc[i] = df1.iloc[i:i+n, 1].sum()
    df2['arrival_time_difference'].loc[i] = df1.iloc[i+n-1, 2] - df1.iloc[i, 2]
    df2['duration'].loc[i] = df1.iloc[i:i+n, 3].sum()

plt.plot(df2['tasks_num'])
plt.xlabel('job chunk')
plt.ylabel('number of tasks')
plt.show()
plt.plot(df2['instances_num'])
plt.xlabel('job chunk')
plt.ylabel('number of instances')
plt.show()
plt.plot(df2['arrival_time_difference'])
plt.xlabel('job chunk')
plt.ylabel('arrival_time_difference')
plt.show()
plt.plot(df2['duration'])
plt.xlabel('job chunk')
plt.ylabel('total duration')
plt.show()
