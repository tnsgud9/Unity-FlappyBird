using TMPro;
using UnityEngine;

/// <summary>
/// ============================================================================
/// 학습 관리자 - 에피소드 관리 및 Epsilon 감소 담당
/// ============================================================================
/// 
/// [주요 역할]
/// 1. 에피소드 관리: 게임오버 시 환경 리셋
/// 2. Epsilon 감소: 학습이 진행될수록 탐색 줄이기
/// 3. [고급] Best 정책 보존: 유전 알고리즘 방식
/// 
/// ============================================================================
/// [에피소드란?]
/// ============================================================================
/// 
/// 에피소드 = 한 번의 게임 플레이
/// - 시작 → 플레이 → 게임오버 → 리셋 → 시작...
/// 
/// 예:
/// - 에피소드 1: 시작 → 2초 생존 → 충돌 → 리셋
/// - 에피소드 2: 시작 → 5초 생존 → 1개 통과 → 충돌 → 리셋
/// - 에피소드 3: 시작 → 10초 생존 → 2개 통과 → 충돌 → 리셋
/// 
/// ============================================================================
/// [Epsilon 감소 원리]
/// ============================================================================
/// 
/// 처음에는 많이 탐색 (ε = 1.0 = 100% 무작위)
/// ↓
/// 점점 줄어듦 (ε = 0.5 = 50% 무작위)
/// ↓
/// 나중에는 거의 활용 (ε = 0.05 = 5% 무작위)
/// 
/// 왜 이렇게 할까?
/// - 처음: 아무것도 모르니까 이것저것 시도 (탐색)
/// - 나중: 뭘 해야 할지 알았으니 학습한 대로 (활용)
/// 
/// ============================================================================
/// [학습 방식]
/// ============================================================================
/// 
/// 통합 학습: 생존 + 갭 중심 유지 + 파이프 통과를 동시에 학습
/// </summary>
public class AgentManager : MonoBehaviour
{
    // ========================================
    // Inspector 설정
    // ========================================
    
    [Header("에이전트 연결")]
    [Tooltip("PlayerAgent 컴포넌트")]
    public PlayerAgent agent;
    
    [Tooltip("학습 정보를 표시할 텍스트")]
    public TextMeshProUGUI infoText;
    
    [Header("학습 속도")]
    [Tooltip("게임 속도: 높을수록 빠른 학습 (1 = 실시간)")]
    public float timeScale = 1f;
    
    [Header("Epsilon 설정")]
    [Tooltip("최소 탐색률: 이 이하로는 내려가지 않음")]
    public float minEpsilon = 0.05f;
    
    [Tooltip("Epsilon이 0이 되는 에피소드 수")]
    public float epsilonDecayEpisodes = 5000f;
    
    [Header("[고급] Best 정책 보존")]
    [Tooltip("이 횟수만큼 개선 없으면 Best Q-Table 복원")]
    public int restoreThreshold = 500;
    
    // ========================================
    // 내부 변수
    // ========================================
    private int _episodeCount;           // 현재 에피소드 수
    private float _bestScore;            // 최고 점수
    private float _initialEpsilon;       // 초기 Epsilon
    private int _noImprovementCount;     // 개선 없는 에피소드 수
    
    // ========================================
    // 초기화
    // ========================================
    void Start()
    {
        // 게임 속도 설정
        Time.timeScale = timeScale;
        
        // 변수 초기화
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
    /// 
    /// 게임오버 시 호출됩니다.
    /// </summary>
    public void EndEpisode()
    {
        float currentScore = agent.Score;
        
        // ========================================
        // [고급] Best 정책 보존
        // ========================================
        // 유전 알고리즘처럼 좋은 정책을 유지합니다.
        // ========================================
        
        if (currentScore > _bestScore)
        {
            // 최고 점수 갱신!
            _bestScore = currentScore;
            
            // 현재 Q-Table 저장
            agent.SaveBestQTable(currentScore);
            
            // 개선 카운터 리셋
            _noImprovementCount = 0;
        }
        else
        {
            // 개선 없음
            _noImprovementCount++;
        }
        
        // ========================================
        // 일정 에피소드 동안 개선 없으면 Best Q-Table 복원
        // ========================================
        // 학습이 잘못된 방향으로 가면 이전 좋은 정책으로 되돌림
        // ========================================
        if (_noImprovementCount >= restoreThreshold)
        {
            // Best Q-Table 복원
            agent.RestoreBestQTable();
            
            // Epsilon 소폭 증가 (재탐색 유도)
            // 기존보다 10% 더 탐색하도록
            agent.SetEpsilon(Mathf.Min(0.5f, agent.GetEpsilon() + 0.1f));
            
            // 개선 카운터 리셋
            _noImprovementCount = 0;
        }
        
        // ========================================
        // Epsilon 감소
        // ========================================
        // 선형 감소: initialEpsilon → minEpsilon
        // ========================================
        
        _episodeCount++;
        
        // 감소 비율 계산
        // 0 에피소드: decayRatio = 0 → epsilon = initialEpsilon
        // 5000 에피소드: decayRatio = 1 → epsilon = 0
        float decayRatio = _episodeCount / epsilonDecayEpisodes;
        
        // 새로운 Epsilon 계산
        float newEpsilon = _initialEpsilon * (1f - decayRatio);
        
        // 최소값 이하로는 내려가지 않음
        newEpsilon = Mathf.Max(minEpsilon, newEpsilon);
        
        // Epsilon 설정
        agent.SetEpsilon(newEpsilon);
        
        // 에피소드 리셋
        ResetEpisode();
    }
    
    /// <summary>
    /// 환경 리셋
    /// 다음 에피소드를 위해 게임 환경을 초기화합니다.
    /// </summary>
    void ResetEpisode()
    {
        // 에이전트 리셋
        agent.Reset();
        
        // 모든 파이프 제거
        DestroyAllPipes();
    }
    
    /// <summary>
    /// 모든 파이프 제거
    /// </summary>
    void DestroyAllPipes()
    {
        // 모든 파이프 찾기
        PipeMovement[] pipes = FindObjectsByType<PipeMovement>(FindObjectsSortMode.None);
        
        // 하나씩 제거
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
