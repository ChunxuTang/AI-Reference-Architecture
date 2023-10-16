# MINIO guide

Label the minio instance
```
kubectl get nodes --show-labels
kubectl label nodes ip-10-0-6-242.ec2.internal minio=true
```
Launch the minio instance
```
kubectl create -f minio-dev.yaml
```
Find the minio endpoint
```
$ kubectl get pods -o custom-columns=POD:.metadata.name,HOSTNAME:.status.hostIP -n minio-dev
POD     HOSTNAME
minio   10.0.6.242
```
Connect to MINIO using
```
s3 = boto3.resource('s3',
                    endpoint_url='http://10.0.6.242:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',
                    verify=False)
```
