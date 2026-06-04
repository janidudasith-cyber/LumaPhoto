using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LumaPhoto;

/// <summary>
/// Per-scene confidence scores produced by Places365-MobileNet inference.
/// All nine weights are normalized so they sum to 1.
/// </summary>
public struct SceneWeights
{
    public float General, Daylight, Portrait, Landscape, Sunset, LowLight, Night, HDR, HighKey;

    /// <summary>Scene type with the highest NN confidence.</summary>
    public readonly SceneType Dominant
    {
        get
        {
            var best = SceneType.General; float max = General;
            Beat(SceneType.Daylight,  Daylight,  ref best, ref max);
            Beat(SceneType.Portrait,  Portrait,  ref best, ref max);
            Beat(SceneType.Landscape, Landscape, ref best, ref max);
            Beat(SceneType.Sunset,    Sunset,    ref best, ref max);
            Beat(SceneType.LowLight,  LowLight,  ref best, ref max);
            Beat(SceneType.Night,     Night,     ref best, ref max);
            Beat(SceneType.HDR,       HDR,       ref best, ref max);
            Beat(SceneType.HighKey,   HighKey,   ref best, ref max);
            return best;
        }
    }

    /// <summary>Confidence in the dominant scene (0–1).</summary>
    public readonly float DominantScore
    {
        get
        {
            float m = General;
            if (Daylight  > m) m = Daylight;
            if (Portrait  > m) m = Portrait;
            if (Landscape > m) m = Landscape;
            if (Sunset    > m) m = Sunset;
            if (LowLight  > m) m = LowLight;
            if (Night     > m) m = Night;
            if (HDR       > m) m = HDR;
            if (HighKey   > m) m = HighKey;
            return m;
        }
    }

    private static void Beat(SceneType t, float v, ref SceneType best, ref float max)
    { if (v > max) { max = v; best = t; } }

    internal void Add(SceneType t, float amount)
    {
        switch (t)
        {
            case SceneType.Daylight:  Daylight  += amount; break;
            case SceneType.Portrait:  Portrait  += amount; break;
            case SceneType.Landscape: Landscape += amount; break;
            case SceneType.Sunset:    Sunset    += amount; break;
            case SceneType.LowLight:  LowLight  += amount; break;
            case SceneType.Night:     Night     += amount; break;
            case SceneType.HDR:       HDR       += amount; break;
            case SceneType.HighKey:   HighKey   += amount; break;
            default:                  General   += amount; break;
        }
    }

    internal void Normalize()
    {
        float total = General + Daylight + Portrait + Landscape + Sunset
                    + LowLight + Night + HDR + HighKey;
        if (total < 1e-6f) { General = 1f; return; }
        float inv = 1f / total;
        General *= inv; Daylight *= inv; Portrait *= inv; Landscape *= inv;
        Sunset  *= inv; LowLight *= inv; Night    *= inv; HDR       *= inv;
        HighKey *= inv;
    }
}

/// <summary>
/// Neural enhancement back-end with two operating modes:
///
/// Mode 1 — Direct Parameter Prediction (preferred, higher quality):
///   Requires "enhancer_params.onnx" next to the exe.
///   Produced by training/train.py on MIT-Adobe FiveK / PPR10K / DPED.
///   Model input : [1, 3, 224, 224] ImageNet-normalised image
///   Model output: [1, 15] enhancement parameters (in slider units)
///
/// Mode 2 — Scene Classification (fallback):
///   Requires "places365_mobilenet.onnx" + "places365_classes.txt".
///   Classifies the image into one of 365 place categories and blends
///   the rule-based parameters toward the NN-preferred scene.
///
/// Both modes degrade gracefully when their model files are absent.
/// </summary>
public sealed class NeuralEnhancer : IDisposable
{
    // ── Mode 1: direct parameter prediction ──────────────────────────────────
    private InferenceSession? _paramSession;
    private string?           _paramInputName;

    // ── Mode 2: Places365 scene classification ────────────────────────────────
    private InferenceSession? _sceneSession;
    private string[]?         _labels;
    private string?           _sceneInputName;

    // Standard ImageNet normalisation (same for both models)
    private static readonly float[] NormMean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] NormStd  = { 0.229f, 0.224f, 0.225f };

    // Number of parameters the trained model outputs
    private const int NumParams = 15;

    /// <summary>True when the trained parameter-prediction model is loaded.</summary>
    public bool HasParamModel => _paramSession != null;

    /// <summary>True when at least one model (param-prediction or scene) is loaded.</summary>
    public bool IsLoaded => HasParamModel || (_sceneSession != null && _labels != null);

    public NeuralEnhancer()
    {
        var dir = AppDomain.CurrentDomain.BaseDirectory;

        // ── Try to load the trained parameter-prediction model first ──────────
        var paramPath = Path.Combine(dir, "enhancer_params.onnx");
        if (File.Exists(paramPath))
        {
            try
            {
                _paramSession   = new InferenceSession(paramPath);
                _paramInputName = _paramSession.InputMetadata.Keys.First();
            }
            catch
            {
                _paramSession?.Dispose();
                _paramSession = null;
            }
        }

        // ── Fallback: Places365 scene classifier ──────────────────────────────
        if (!HasParamModel)
        {
            var scenePath = Path.Combine(dir, "places365_mobilenet.onnx");
            var lblPath   = Path.Combine(dir, "places365_classes.txt");
            if (File.Exists(scenePath) && File.Exists(lblPath))
            {
                try
                {
                    _sceneSession   = new InferenceSession(scenePath);
                    _sceneInputName = _sceneSession.InputMetadata.Keys.First();
                    _labels         = ParseLabels(File.ReadAllLines(lblPath));
                }
                catch
                {
                    _sceneSession?.Dispose();
                    _sceneSession = null;
                    _labels       = null;
                }
            }
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Mode 1: run the trained model and return enhancement parameters directly.
    /// Runs inference on the full image and a center 80% crop, then averages the
    /// predictions — improves accuracy for portraits and centered subjects.
    /// Returns null when the param model is not loaded or inference fails.
    /// </summary>
    public AdjustmentState? PredictParams(byte[] bgra, int width, int height)
    {
        if (!HasParamModel) return null;
        try
        {
            // Full-image prediction
            var p0 = RunParamInference(PreprocessRegion(bgra, width, height, 0, 0, width, height));
            if (p0 == null) return null;

            // Center 80% crop — improves portrait / centered-subject accuracy
            int cx = (int)(width  * 0.1f);
            int cy = (int)(height * 0.1f);
            int cw = (int)(width  * 0.8f);
            int ch = (int)(height * 0.8f);
            var p1 = RunParamInference(PreprocessRegion(bgra, width, height, cx, cy, cw, ch));
            if (p1 == null) return p0;

            // Average predictions; keep vignette from full-image only (it depends on borders)
            return new AdjustmentState
            {
                Exposure   = (p0.Exposure   + p1.Exposure)   * 0.5f,
                Brilliance = (p0.Brilliance + p1.Brilliance) * 0.5f,
                Highlights = (p0.Highlights + p1.Highlights) * 0.5f,
                Shadows    = (p0.Shadows    + p1.Shadows)    * 0.5f,
                Contrast   = (p0.Contrast   + p1.Contrast)   * 0.5f,
                Brightness = (p0.Brightness + p1.Brightness) * 0.5f,
                BlackPoint = (p0.BlackPoint + p1.BlackPoint) * 0.5f,
                Saturation = (p0.Saturation + p1.Saturation) * 0.5f,
                Vibrance   = (p0.Vibrance   + p1.Vibrance)   * 0.5f,
                Warmth     = (p0.Warmth     + p1.Warmth)     * 0.5f,
                Tint       = (p0.Tint       + p1.Tint)       * 0.5f,
                Sharpness  = (p0.Sharpness  + p1.Sharpness)  * 0.5f,
                Definition = (p0.Definition + p1.Definition) * 0.5f,
                Noise      = (p0.Noise      + p1.Noise)      * 0.5f,
                Vignette   = p0.Vignette,  // border-dependent; use full-image only
            };
        }
        catch { return null; }
    }

    private AdjustmentState? RunParamInference(DenseTensor<float> tensor)
    {
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(_paramInputName!, tensor) };
        using var results = _paramSession!.Run(inputs);
        var raw = results[0].AsEnumerable<float>().ToArray();
        return raw.Length >= NumParams ? RawToState(raw) : null;
    }

    /// <summary>
    /// Mode 2: run Places365 scene classification and return scene confidence weights.
    /// Returns null when the scene model is not loaded or inference fails.
    /// </summary>
    public SceneWeights? Analyze(byte[] bgra, int width, int height)
    {
        if (_sceneSession == null || _labels == null) return null;
        try
        {
            var tensor = PreprocessRegion(bgra, width, height, 0, 0, width, height);
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_sceneInputName!, tensor) };
            using var results = _sceneSession!.Run(inputs);
            var probs = Softmax(results[0].AsEnumerable<float>().ToArray());
            return BuildWeights(probs);
        }
        catch { return null; }
    }

    // ── Shared preprocessing ──────────────────────────────────────────────────

    /// <summary>
    /// Resize a rectangular region of the source image to 224×224 using bilinear
    /// interpolation with half-pixel alignment (same convention as PyTorch's
    /// F.interpolate with align_corners=False).
    /// </summary>
    private static DenseTensor<float> PreprocessRegion(
        byte[] bgra, int srcW, int srcH,
        int rx, int ry, int rw, int rh)
    {
        const int sz = 224;
        var t = new DenseTensor<float>(new[] { 1, 3, sz, sz });

        for (int y = 0; y < sz; y++)
        for (int x = 0; x < sz; x++)
        {
            // Half-pixel center convention matches PyTorch's align_corners=False
            float fx = rx + ((x + 0.5f) / sz) * rw - 0.5f;
            float fy = ry + ((y + 0.5f) / sz) * rh - 0.5f;

            int x0 = Math.Clamp((int)MathF.Floor(fx),     0, srcW - 1);
            int x1 = Math.Clamp((int)MathF.Floor(fx) + 1, 0, srcW - 1);
            int y0 = Math.Clamp((int)MathF.Floor(fy),     0, srcH - 1);
            int y1 = Math.Clamp((int)MathF.Floor(fy) + 1, 0, srcH - 1);
            float wx = fx - MathF.Floor(fx);
            float wy = fy - MathF.Floor(fy);

            float Px(int px, int py, int ch) => bgra[(py * srcW + px) * 4 + ch];

            float r = (1-wx)*(1-wy)*Px(x0,y0,2) + wx*(1-wy)*Px(x1,y0,2)
                    + (1-wx)*wy*Px(x0,y1,2)      + wx*wy*Px(x1,y1,2);
            float g = (1-wx)*(1-wy)*Px(x0,y0,1) + wx*(1-wy)*Px(x1,y0,1)
                    + (1-wx)*wy*Px(x0,y1,1)      + wx*wy*Px(x1,y1,1);
            float b = (1-wx)*(1-wy)*Px(x0,y0,0) + wx*(1-wy)*Px(x1,y0,0)
                    + (1-wx)*wy*Px(x0,y1,0)      + wx*wy*Px(x1,y1,0);

            t[0, 0, y, x] = (r / 255f - NormMean[0]) / NormStd[0];
            t[0, 1, y, x] = (g / 255f - NormMean[1]) / NormStd[1];
            t[0, 2, y, x] = (b / 255f - NormMean[2]) / NormStd[2];
        }
        return t;
    }

    // ── Mode 1 helpers ────────────────────────────────────────────────────────

    private static AdjustmentState RawToState(float[] raw) => new AdjustmentState
    {
        // Index order must match PARAM_NAMES in training/pipeline.py
        Exposure   = Math.Clamp(raw[0],  -100, 100),
        Brilliance = Math.Clamp(raw[1],  -100, 100),
        Highlights = Math.Clamp(raw[2],  -100, 100),
        Shadows    = Math.Clamp(raw[3],  -100, 100),
        Contrast   = Math.Clamp(raw[4],  -100, 100),
        Brightness = Math.Clamp(raw[5],  -100, 100),
        BlackPoint = Math.Clamp(raw[6],  -100, 100),
        Saturation = Math.Clamp(raw[7],  -100, 100),
        Vibrance   = Math.Clamp(raw[8],  -100, 100),
        Warmth     = Math.Clamp(raw[9],  -100, 100),
        Tint       = Math.Clamp(raw[10], -100, 100),
        Sharpness  = Math.Clamp(raw[11],    0, 100),
        Definition = Math.Clamp(raw[12],    0, 100),
        Noise      = Math.Clamp(raw[13],    0, 100),
        Vignette   = Math.Clamp(raw[14],    0, 100),
    };

    // ── Mode 2 helpers ────────────────────────────────────────────────────────

    private static float[] Softmax(float[] logits)
    {
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
        float sum = 0f;
        var e = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++) { e[i] = MathF.Exp(logits[i] - max); sum += e[i]; }
        for (int i = 0; i < e.Length;      i++) e[i] /= sum;
        return e;
    }

    private SceneWeights BuildWeights(float[] probs)
    {
        var w = new SceneWeights();
        for (int i = 0; i < probs.Length && i < _labels!.Length; i++)
        {
            if (probs[i] < 0.004f) continue;
            if (CategoryMap.TryGetValue(_labels[i], out var entry))
                w.Add(entry.scene, probs[i] * entry.boost);
        }
        w.Normalize();
        return w;
    }

    // Parse "/a/forest/broadleaf 127" → "forest_broadleaf"
    private static string[] ParseLabels(string[] lines)
    {
        var result = new string[lines.Length];
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();
            int sp = line.LastIndexOf(' ');
            if (sp > 0) line = line[..sp];
            int sl = line.IndexOf('/', 1);
            if (sl >= 0) line = line[(sl + 1)..];
            result[i] = line.Replace('/', '_');
        }
        return result;
    }

    public void Dispose()
    {
        _paramSession?.Dispose();
        _sceneSession?.Dispose();
    }

    // ── Places365 category → (SceneType, boost) mapping ──────────────────────
    // Category names are normalised: /a/ prefix stripped, remaining / → _.
    // Unmapped categories (offices, kitchens, etc.) contribute to General.
    private static readonly Dictionary<string, (SceneType scene, float boost)> CategoryMap =
        new(StringComparer.OrdinalIgnoreCase)
    {
        // Night
        ["discotheque"]               = (SceneType.Night,    1.0f),
        ["nightclub"]                 = (SceneType.Night,    1.0f),
        ["concert_hall"]              = (SceneType.Night,    0.7f),
        ["pub_indoor"]                = (SceneType.Night,    0.5f),
        ["casino_indoor"]             = (SceneType.Night,    0.7f),
        ["movie_theater_indoor"]      = (SceneType.Night,    0.6f),
        ["amusement_arcade"]          = (SceneType.Night,    0.5f),
        ["karaoke_room"]              = (SceneType.Night,    0.5f),

        // LowLight
        ["basement"]                  = (SceneType.LowLight, 1.0f),
        ["corridor"]                  = (SceneType.LowLight, 0.8f),
        ["hallway"]                   = (SceneType.LowLight, 0.7f),
        ["home_theater"]              = (SceneType.LowLight, 0.7f),
        ["sauna"]                     = (SceneType.LowLight, 0.7f),
        ["parking_garage_indoor"]     = (SceneType.LowLight, 0.8f),
        ["parking_lot"]               = (SceneType.LowLight, 0.3f),
        ["bedroom"]                   = (SceneType.LowLight, 0.6f),
        ["dorm_room"]                 = (SceneType.LowLight, 0.6f),
        ["hotel_room"]                = (SceneType.LowLight, 0.5f),
        ["television_room"]           = (SceneType.LowLight, 0.5f),
        ["bar"]                       = (SceneType.LowLight, 0.6f),
        ["beer_hall"]                 = (SceneType.LowLight, 0.5f),
        ["library_indoor"]            = (SceneType.LowLight, 0.5f),
        ["restaurant"]                = (SceneType.LowLight, 0.3f),
        ["dining_room"]               = (SceneType.LowLight, 0.4f),
        ["living_room"]               = (SceneType.LowLight, 0.4f),
        ["diner_indoor"]              = (SceneType.LowLight, 0.3f),
        ["gas_station"]               = (SceneType.LowLight, 0.3f),
        ["tunnel"]                    = (SceneType.LowLight, 0.7f),
        ["subway_station_platform"]   = (SceneType.LowLight, 0.5f),

        // HDR
        ["cave"]                      = (SceneType.HDR,      1.0f),
        ["grotto"]                    = (SceneType.HDR,      1.0f),
        ["canyon"]                    = (SceneType.HDR,      0.9f),
        ["crevasse"]                  = (SceneType.HDR,      0.8f),
        ["church_indoor"]             = (SceneType.HDR,      0.7f),
        ["nave"]                      = (SceneType.HDR,      0.8f),
        ["arch"]                      = (SceneType.HDR,      0.5f),
        ["mausoleum"]                 = (SceneType.HDR,      0.6f),
        ["catacomb"]                  = (SceneType.HDR,      0.8f),
        ["aqueduct"]                  = (SceneType.HDR,      0.5f),

        // HighKey
        ["snowfield"]                 = (SceneType.HighKey,  1.0f),
        ["ski_slope"]                 = (SceneType.HighKey,  0.9f),
        ["igloo"]                     = (SceneType.HighKey,  0.8f),
        ["mountain_snowy"]            = (SceneType.HighKey,  0.7f),
        ["ice_shelf"]                 = (SceneType.HighKey,  0.7f),
        ["ice_skating_rink_outdoor"]  = (SceneType.HighKey,  0.6f),
        ["ice_skating_rink_indoor"]   = (SceneType.HighKey,  0.5f),
        ["glacier"]                   = (SceneType.HighKey,  0.6f),
        ["sky"]                       = (SceneType.HighKey,  0.4f),
        ["hospital_room"]             = (SceneType.HighKey,  0.5f),
        ["operating_room"]            = (SceneType.HighKey,  0.7f),

        // Landscape
        ["forest_broadleaf"]          = (SceneType.Landscape, 1.0f),
        ["forest_needleleaf"]         = (SceneType.Landscape, 1.0f),
        ["forest_path"]               = (SceneType.Landscape, 0.9f),
        ["bamboo_forest"]             = (SceneType.Landscape, 1.0f),
        ["rainforest"]                = (SceneType.Landscape, 1.0f),
        ["jungle"]                    = (SceneType.Landscape, 1.0f),
        ["mountain"]                  = (SceneType.Landscape, 1.0f),
        ["mountain_path"]             = (SceneType.Landscape, 0.9f),
        ["ocean"]                     = (SceneType.Landscape, 1.0f),
        ["river"]                     = (SceneType.Landscape, 1.0f),
        ["waterfall"]                 = (SceneType.Landscape, 1.0f),
        ["lake_natural"]              = (SceneType.Landscape, 1.0f),
        ["pond"]                      = (SceneType.Landscape, 0.8f),
        ["valley"]                    = (SceneType.Landscape, 1.0f),
        ["coast"]                     = (SceneType.Landscape, 0.9f),
        ["lagoon"]                    = (SceneType.Landscape, 0.9f),
        ["islet"]                     = (SceneType.Landscape, 0.9f),
        ["bayou"]                     = (SceneType.Landscape, 0.9f),
        ["swamp"]                     = (SceneType.Landscape, 0.9f),
        ["marsh"]                     = (SceneType.Landscape, 0.9f),
        ["wetland"]                   = (SceneType.Landscape, 0.9f),
        ["creek"]                     = (SceneType.Landscape, 0.9f),
        ["cliff"]                     = (SceneType.Landscape, 0.9f),
        ["butte"]                     = (SceneType.Landscape, 0.9f),
        ["volcano"]                   = (SceneType.Landscape, 0.9f),
        ["tundra"]                    = (SceneType.Landscape, 0.9f),
        ["savanna"]                   = (SceneType.Landscape, 1.0f),
        ["beach"]                     = (SceneType.Landscape, 0.8f),
        ["desert_sand"]               = (SceneType.Landscape, 0.9f),
        ["desert_vegetation"]         = (SceneType.Landscape, 0.9f),
        ["desert_road"]               = (SceneType.Landscape, 0.7f),
        ["field_wild"]                = (SceneType.Landscape, 0.9f),
        ["field_cultivated"]          = (SceneType.Landscape, 0.8f),
        ["hayfield"]                  = (SceneType.Landscape, 0.9f),
        ["corn_field"]                = (SceneType.Landscape, 0.8f),
        ["wheat_field"]               = (SceneType.Landscape, 0.9f),
        ["rice_paddy"]                = (SceneType.Landscape, 0.9f),
        ["vineyard"]                  = (SceneType.Landscape, 0.9f),
        ["orchard"]                   = (SceneType.Landscape, 0.8f),
        ["pasture"]                   = (SceneType.Landscape, 0.8f),
        ["tree_farm"]                 = (SceneType.Landscape, 0.8f),
        ["wind_farm"]                 = (SceneType.Landscape, 0.7f),
        ["dam"]                       = (SceneType.Landscape, 0.7f),
        ["iceberg"]                   = (SceneType.Landscape, 0.8f),
        ["hot_spring"]                = (SceneType.Landscape, 0.8f),
        ["watering_hole"]             = (SceneType.Landscape, 0.8f),
        ["wave"]                      = (SceneType.Landscape, 0.9f),
        ["harbor"]                    = (SceneType.Landscape, 0.7f),
        ["lighthouse"]                = (SceneType.Landscape, 0.8f),
        ["moat"]                      = (SceneType.Landscape, 0.7f),

        // Sunset — Places365 has no explicit sunset category; we get weak
        // signal from golden-hour outdoor spaces. Portrait/pixel analysis
        // remains the primary detector.
        ["pier"]                      = (SceneType.Sunset,   0.3f),
        ["boardwalk"]                 = (SceneType.Sunset,   0.3f),
        ["beach_house"]               = (SceneType.Sunset,   0.2f),
        ["waterfront"]                = (SceneType.Sunset,   0.3f),

        // Portrait
        ["beauty_salon"]              = (SceneType.Portrait, 0.6f),
        ["dressing_room"]             = (SceneType.Portrait, 0.5f),
        ["nursery"]                   = (SceneType.Portrait, 0.4f),
        ["wedding_reception"]         = (SceneType.Portrait, 0.7f),

        // Daylight
        ["park"]                      = (SceneType.Daylight, 0.8f),
        ["botanical_garden"]          = (SceneType.Daylight, 0.8f),
        ["formal_garden"]             = (SceneType.Daylight, 0.8f),
        ["cottage_garden"]            = (SceneType.Daylight, 0.8f),
        ["japanese_garden"]           = (SceneType.Daylight, 0.8f),
        ["herb_garden"]               = (SceneType.Daylight, 0.7f),
        ["topiary_garden"]            = (SceneType.Daylight, 0.8f),
        ["roof_garden"]               = (SceneType.Daylight, 0.7f),
        ["vegetable_garden"]          = (SceneType.Daylight, 0.7f),
        ["golf_course"]               = (SceneType.Daylight, 0.9f),
        ["playground"]                = (SceneType.Daylight, 0.8f),
        ["picnic_area"]               = (SceneType.Daylight, 0.8f),
        ["campsite"]                  = (SceneType.Daylight, 0.7f),
        ["baseball_field"]            = (SceneType.Daylight, 0.9f),
        ["football_field"]            = (SceneType.Daylight, 0.9f),
        ["tennis_court_outdoor"]      = (SceneType.Daylight, 0.9f),
        ["volleyball_court_outdoor"]  = (SceneType.Daylight, 0.9f),
        ["racecourse"]                = (SceneType.Daylight, 0.8f),
        ["courtyard"]                 = (SceneType.Daylight, 0.6f),
        ["patio"]                     = (SceneType.Daylight, 0.7f),
        ["amusement_park"]            = (SceneType.Daylight, 0.7f),
        ["farm"]                      = (SceneType.Daylight, 0.7f),
        ["residential_neighborhood"]  = (SceneType.Daylight, 0.5f),
        ["swimming_pool_outdoor"]     = (SceneType.Daylight, 0.8f),
        ["market_outdoor"]            = (SceneType.Daylight, 0.7f),
        ["marketplace_outdoor"]       = (SceneType.Daylight, 0.7f),
        ["plaza"]                     = (SceneType.Daylight, 0.6f),
        ["town_square"]               = (SceneType.Daylight, 0.6f),
        ["promenade"]                 = (SceneType.Daylight, 0.6f),
        ["soccer_field"]              = (SceneType.Daylight, 0.9f),
        ["track_outdoor"]             = (SceneType.Daylight, 0.8f),
        ["ski_resort"]                = (SceneType.Daylight, 0.7f),
    };
}
