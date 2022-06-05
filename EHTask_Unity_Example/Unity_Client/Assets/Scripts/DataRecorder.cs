using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class DataRecorder : MonoBehaviour
{
    public Camera headCamera;
    // the number of data item in one recording
    public int dataNumber = 250;
    // time rate to sample the data (hz)
    float sampleRate = 25;
    // time offset (ms) used to correct the sampling time, it depends on how fast your machine runs
    float timeOffset = 2;
    // Eye gaze data on the screen
    Queue<string> gazeData;
    // head velocity data
    Queue<string> headData;
    public string recordingsString;
    public bool Running;

    // Start is called before the first frame update
    void Start()
    {
        gazeData = new Queue<string>();
        headData = new Queue<string>();
        recordingsString = null;
        Running = true;
        StartCoroutine(RecordData());
    }


    IEnumerator RecordData()
    {
        WaitForSecondsRealtime waitTime = new WaitForSecondsRealtime(1f/sampleRate - timeOffset/1000);

        while (Running)
        {
            SampleData();
            yield return waitTime;
        }
    }

    void SampleData()
    {
        // Timestamp
        System.TimeSpan timeSpan = System.DateTime.Now - new System.DateTime(1970, 1, 1, 0, 0, 0);
        long time = (long)timeSpan.TotalMilliseconds - 8 * 60 * 60 * 1000;
        string timeStamp = time.ToString();


        // Eye Gaze Data
        // In real applications, get the gaze data from your eye tracker
        // (0, 0) at Bottom-left, (1, 1) at Top-right
        float gazeX = Random.value;
        float gazeY = Random.value;
        string gazeInfo = gazeX.ToString("f2") + "," + gazeY.ToString("f2");

        // Head Rotation Velocity
        float headVelX = headCamera.GetComponent<CalculateHeadVelocity>().headVelX;
        float headVelY = headCamera.GetComponent<CalculateHeadVelocity>().headVelY;
        string headInfo = headVelX.ToString("f2") + "," + headVelY.ToString("f2");
        
        if (gazeData.Count < dataNumber)
        {
            gazeData.Enqueue(gazeInfo);
            headData.Enqueue(headInfo);
            if (gazeData.Count == dataNumber)
            {
                recordingsString = timeStamp;
                foreach (string data in gazeData)
                    recordingsString = recordingsString + "," + data;
                foreach (string data in headData)
                    recordingsString = recordingsString + "," + data;
            }
        }
        else if (gazeData.Count == dataNumber)
        {
            gazeData.Dequeue();         
            headData.Dequeue();
            gazeData.Enqueue(gazeInfo);
            headData.Enqueue(headInfo);
            recordingsString = timeStamp;
            foreach (string data in gazeData)
                recordingsString = recordingsString + "," + data;
            foreach (string data in headData)
                recordingsString = recordingsString + "," + data;
        }
    }


}
