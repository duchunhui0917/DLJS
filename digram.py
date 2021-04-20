import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./jobs.csv')
df = df.sort_values(by='submit_time', ascending=True)
df.to_csv('./sort_jobs.csv')
# size = df['cpu'].rank()
# plt.scatter(df['submit_time'], df['instances_num'], s=size)
# plt.show()

# plot the number of tasks with submit time
# df1 = df.groupby('submit_time').size()
# plt.xlabel('submit time')
# plt.ylabel('number of tasks')
# plt.plot(df1)
# plt.show()
#
# df2 = df.groupby('submit_time')['job_id'].mean()
# df2 = df2.groupby('submit_time').size()
# plt.scatter(df2.index, df2)
# plt.show()
