using UnityEngine;

public class PipeSpawner : MonoBehaviour
{
    public GameObject PipePrefab;
    public float SpawnRateSec = 2f;
    public float RandomYMaxOffset = 10f;
    public float RandomYMinOffset = 0f;
    private float _timer = int.MaxValue;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        _timer += Time.deltaTime;

        if (_timer > SpawnRateSec)
        {
            // TODO: 랜덤하게 y축을 바꿔서 생성하도록 해봅시다.
            // HINT Random.Range
            // Pipe 생성한 코드를 작성
            Instantiate(PipePrefab, new Vector3(10,Random.Range(RandomYMinOffset,RandomYMaxOffset),0),Quaternion.identity);
            Debug.Log("파이프 생성");
            _timer = 0f;
        }
        // Debug.Log(_timer);
    }
}
