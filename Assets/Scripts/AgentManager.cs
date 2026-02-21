using TMPro;
using UnityEngine;

/// <summary>
/// 학습 관리자 - 에피소드 관리 및 Epsilon 감소 담당
/// 
/// [주요 역할]
/// 1. 에피소드 관리: 게임오버 시 환경 리셋
/// 2. Epsilon 감소: 학습이 진행될수록 탐색 줄이기
/// 3. [고급] Best 정책 보존: 유전 알고리즘 방식으로 좋은 정책 유지
/// 
/// [Epsilon 감소 원리]
/// - 초기: ε=1.0 (100% 무작위 탐색)
/// - 학습 진행: ε 점진적 감소
/// - 최종: ε=0.05 (5% 탐색, 95% 활용)
/// </summary>
public class AgentManager : MonoBehaviour
{
    // ========================================
    // Inspector 설정
    // ========================================
    
    [Header("에이전트 연결")]
    public PlayerAgent agent;
    public TextMeshProUGUI infoText;
    
    [Header("학습 속도")]
    public float timeScale = 1f;              // 게임 속도 (높을수록 빠른 학습)
    
    [Header("Epsilon 설정")]
    public float minEpsilon = 0.05f;          // 최소 탐색률
    public float epsilonDecayEpisodes = 5000f; // 감소 완료까지의 에피소드 수
    
    [Header("[고급] Best 정책 보존")]
    public int restoreThreshold = 500;        // 이만큼 개선 없으면 복원
    
    // ========================================
    // 내부 변수
    // ========================================
    private int _episodeCount;
    private float _bestScore;
    private float _initialEpsilon;
    private int _noImprovementCount;
    
    // ========================================
    // 초기화
    // ========================================
    void Start()
    {
        Time.timeScale = timeScale;
        _episodeCount = 0;
        _bestScore = 0;
        _initialEpsilon = agent.GetEpsilon();
        _noImprovementCount = 0;
    }
    
    // ========================================
    // UI 업데이트
    // ========================================
    void Update()
    {
        if (infoText != null)
        {
            infoText.text = $"Score: {agent.Score}\n" +
                            $"Best: {_bestScore:F0}\n" +
                            $"Episode: {_episodeCount}\n" +
                            $"Epsilon: {agent.GetEpsilon():F2}";
        }
    }
    
    // ========================================
    // 에피소드 관리
    // ========================================
    
    /// <summary>
    /// 에피소드 종료 처리
    /// </summary>
    public void EndEpisode()
    {
        float currentScore = agent.Score;
        
        // ========================================
        // [고급] Best 정책 보존
        // ========================================
        
        if (currentScore > _bestScore)
        {
            // 최고 점수 갱신: Q-Table 저장
            _bestScore = currentScore;
            agent.SaveBestQTable(currentScore);
            _noImprovementCount = 0;
        }
        else
        {
            // 개선 없음
            _noImprovementCount++;
        }
        
        // 일정 에피소드 동안 개선 없으면 Best Q-Table 복원
        if (_noImprovementCount >= restoreThreshold)
        {
            agent.RestoreBestQTable();
            
            // Epsilon 소폭 증가 (재탐색 유도)
            agent.SetEpsilon(Mathf.Min(0.5f, agent.GetEpsilon() + 0.1f));
            _noImprovementCount = 0;
        }
        
        // ========================================
        // Epsilon 감소
        // ========================================
        
        _episodeCount++;
        
        // 선형 감소: initialEpsilon → minEpsilon
        float decayRatio = _episodeCount / epsilonDecayEpisodes;
        float newEpsilon = Mathf.Max(minEpsilon, _initialEpsilon * (1f - decayRatio));
        agent.SetEpsilon(newEpsilon);
        
        // 에피소드 리셋
        ResetEpisode();
    }
    
    /// <summary>
    /// 환경 리셋: 에이전트 위치 초기화 + 파이프 제거
    /// </summary>
    void ResetEpisode()
    {
        agent.Reset();
        DestroyAllPipes();
    }
    
    /// <summary>
    /// 모든 파이프 제거
    /// </summary>
    void DestroyAllPipes()
    {
        PipeMovement[] pipes = FindObjectsByType<PipeMovement>(FindObjectsSortMode.None);
        foreach (var pipe in pipes)
            Destroy(pipe.gameObject);
    }
    
    // ========================================
    // 유틸리티
    // ========================================
    
    /// <summary>
    /// 게임 속도 설정
    /// </summary>
    public void SetTimeScale(float value)
    {
        timeScale = value;
        Time.timeScale = timeScale;
    }
}
