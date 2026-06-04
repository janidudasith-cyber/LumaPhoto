using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Runtime.CompilerServices;

namespace LumaPhoto;

public enum FilterType
{
    None, Vivid, Warm, Cool, Sunset, TealOrange, Process, Lush, Retro, Instant,
    Fade, ZoomBlur, Grainy, Gritty,
    Chrome, Sepia, Silvertone, Matte, Dramatic, Mono, Noir
}

// Scene type detected during analysis — drives auto enhance decisions
public enum SceneType
{
    General,    // well-lit, no dominant cue
    Daylight,   // bright outdoor / good-light scene
    Portrait,   // significant skin-tone presence
    Landscape,  // sky or vegetation dominant
    Sunset,     // warm-toned, mid-brightness
    LowLight,   // moderately dark (dim room, overcast, shade)
    Night,      // very dark with bright hotspots (streetlights, stars)
    HDR,        // high dynamic range — both dark & bright zones simultaneously
    HighKey     // very bright, minimal shadows (studio, snow, beach)
}

public class AdjustmentState
{
    public float Exposure    = 0;
    public float Brilliance  = 0;
    public float Highlights  = 0;
    public float Shadows     = 0;
    public float Contrast    = 0;
    public float Brightness  = 0;
    public float BlackPoint  = 0;
    public float Saturation  = 0;
    public float Vibrance    = 0;
    public float Warmth      = 0;
    public float Tint        = 0;
    public float Sharpness   = 0;
    public float Definition  = 0;
    public float Noise       = 0;
    public float Vignette    = 0;
    public float RotateZ     = 0;   // degrees, -45 to 45
    public float TiltX       = 0;   // perspective vertical, -100 to 100
    public float TiltY       = 0;   // perspective horizontal, -100 to 100

    public FilterType Filter          = FilterType.None;
    public float      FilterIntensity = 100f;

    // ── Tone curve — 5 control points: output value at input 0, 64, 128, 192, 255 ──
    public float Curve0   = 0f;
    public float Curve64  = 64f;
    public float Curve128 = 128f;
    public float Curve192 = 192f;
    public float Curve255 = 255f;

    // ── HSL Mixer — hue shift (-100..+100 maps to -180°..+180°) and saturation
    //    per color range: Red, Orange, Yellow, Green, Cyan, Blue, Purple ──
    public float HslRH = 0, HslRS = 0;
    public float HslOH = 0, HslOS = 0;
    public float HslYH = 0, HslYS = 0;
    public float HslGH = 0, HslGS = 0;
    public float HslCH = 0, HslCS = 0;
    public float HslBH = 0, HslBS = 0;
    public float HslPH = 0, HslPS = 0;

    public bool HasCurve =>
        Curve0 != 0 || Curve64 != 64 || Curve128 != 128 || Curve192 != 192 || Curve255 != 255;

    public bool HasHsl =>
        HslRH != 0 || HslRS != 0 || HslOH != 0 || HslOS != 0 ||
        HslYH != 0 || HslYS != 0 || HslGH != 0 || HslGS != 0 ||
        HslCH != 0 || HslCS != 0 || HslBH != 0 || HslBS != 0 ||
        HslPH != 0 || HslPS != 0;

    public void Reset()
    {
        Exposure = Brilliance = Highlights = Shadows = Contrast = Brightness = 0;
        BlackPoint = Saturation = Vibrance = Warmth = Tint = 0;
        Sharpness = Definition = Noise = Vignette = 0;
        RotateZ = TiltX = TiltY = 0;
        Filter = FilterType.None;
        FilterIntensity = 100f;
        Curve0 = 0; Curve64 = 64; Curve128 = 128; Curve192 = 192; Curve255 = 255;
        HslRH = HslRS = HslOH = HslOS = HslYH = HslYS = 0;
        HslGH = HslGS = HslCH = HslCS = HslBH = HslBS = HslPH = HslPS = 0;
    }

    public AdjustmentState Clone() => (AdjustmentState)MemberwiseClone();

    /// <summary>Copy every field from <paramref name="o"/> into this instance.
    /// Used to restore history snapshots in-place (since _adj is readonly).</summary>
    public void CopyFrom(AdjustmentState o)
    {
        Exposure = o.Exposure; Brilliance = o.Brilliance; Highlights = o.Highlights;
        Shadows = o.Shadows; Contrast = o.Contrast; Brightness = o.Brightness;
        BlackPoint = o.BlackPoint; Saturation = o.Saturation; Vibrance = o.Vibrance;
        Warmth = o.Warmth; Tint = o.Tint; Sharpness = o.Sharpness; Definition = o.Definition;
        Noise = o.Noise; Vignette = o.Vignette;
        RotateZ = o.RotateZ; TiltX = o.TiltX; TiltY = o.TiltY;
        Filter = o.Filter; FilterIntensity = o.FilterIntensity;
        Curve0 = o.Curve0; Curve64 = o.Curve64; Curve128 = o.Curve128;
        Curve192 = o.Curve192; Curve255 = o.Curve255;
        HslRH = o.HslRH; HslRS = o.HslRS; HslOH = o.HslOH; HslOS = o.HslOS;
        HslYH = o.HslYH; HslYS = o.HslYS; HslGH = o.HslGH; HslGS = o.HslGS;
        HslCH = o.HslCH; HslCS = o.HslCS; HslBH = o.HslBH; HslBS = o.HslBS;
        HslPH = o.HslPH; HslPS = o.HslPS;
    }
}

public class ImageAnalysis
{
    public float LMean, Contrast, DarkRatio, BrightRatio, RMean, GMean, BMean;
    public float LStdDev;        // luminance std dev — measures dynamic range
    public float SkinRatio;      // fraction of pixels that look like skin tone
    public float SkyRatio;       // fraction of pixels that look like sky (blue-dominant, bright)
    public float GreenRatio;     // fraction of pixels that are green-dominant (vegetation)
    public float WarmRatio;      // fraction of pixels that are red/orange dominant
    public float HotspotRatio;   // fraction of pixels above lum 230 — bright hotspots (night lights, stars)
    public SceneType Scene;
}

public static class ImageProcessor
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Clamp(float v, float lo = 0, float hi = 255) =>
        v < lo ? lo : v > hi ? hi : v;

    // ── Deep image analysis ──
    public static ImageAnalysis Analyze(byte[] pixels, int width, int height)
    {
        // Sample every Nth pixel for speed — at least ~40000 samples
        int step = Math.Max(1, (int)Math.Sqrt((double)width * height / 40000.0));

        double rSum = 0, gSum = 0, bSum = 0, lSum = 0;
        int dark = 0, bright = 0, hotspot = 0, count = 0;
        int skin = 0, sky = 0, green = 0, warm = 0;

        for (int y = 0; y < height; y += step)
        for (int x = 0; x < width;  x += step)
        {
            int i = (y * width + x) * 4;
            float r = pixels[i + 2], g = pixels[i + 1], b = pixels[i];
            float lum = r * .299f + g * .587f + b * .114f;
            rSum += r; gSum += g; bSum += b; lSum += lum;
            count++;

            if (lum < 55)       dark++;
            else if (lum > 200) bright++;
            if (lum > 230)      hotspot++; // very bright points — stars, streetlights, candles

            // Skin tone: R > G > B, specific R/G ratio, mid-brightness.
            // Upper rg cap tightened to 1.45 (was 1.65) so sandy/earthy warm tones
            // (rg ≈ 1.5–1.7) no longer register as skin.
            if (r > 100 && r > g && g > b && r > b * 1.15f &&
                (r / Math.Max(g, 1f)) is float rg && rg >= 1.05f && rg <= 1.45f &&
                lum > 60 && lum < 210)
                skin++;

            // Sky: blue dominant, bright
            if (b > r * 1.1f && b > g * 1.05f && lum > 100)
                sky++;

            // Green vegetation
            if (g > r * 1.1f && g > b * 1.1f && lum > 40)
                green++;

            // Warm (sunset/fire tones): very red/orange
            if (r > 180 && r > g * 1.3f && r > b * 1.8f)
                warm++;
        }

        float rMean = (float)(rSum / count), gMean = (float)(gSum / count),
              bMean = (float)(bSum / count), lMean = (float)(lSum / count);

        // Variance pass for contrast + lum std dev
        double lVar = 0, rVar = 0, gVar = 0, bVar = 0;
        for (int y = 0; y < height; y += step)
        for (int x = 0; x < width;  x += step)
        {
            int i = (y * width + x) * 4;
            float r = pixels[i + 2] - rMean,
                  g = pixels[i + 1] - gMean,
                  bv = pixels[i]    - bMean;
            float lv = (pixels[i+2] * .299f + pixels[i+1] * .587f + pixels[i] * .114f) - lMean;
            rVar += r * r; gVar += g * g; bVar += bv * bv; lVar += lv * lv;
        }

        float contrast = (float)(Math.Sqrt(rVar / count) + Math.Sqrt(gVar / count) + Math.Sqrt(bVar / count)) / 3f;
        float lStdDev  = (float)Math.Sqrt(lVar / count);

        float skinRatio    = (float)skin    / count;
        float skyRatio     = (float)sky     / count;
        float greenRatio   = (float)green   / count;
        float warmRatio    = (float)warm    / count;
        float darkRatio    = (float)dark    / count;
        float brightRatio  = (float)bright  / count;
        float hotspotRatio = (float)hotspot / count;

        // ── Scene classification — evaluated in priority order ──
        SceneType scene;

        // Night: very dark overall, but tiny fraction of extremely bright hotspots
        // (streetlights, stars, windows at night)
        if (lMean < 42 && darkRatio > 0.55f && hotspotRatio > 0.004f && hotspotRatio < 0.18f)
            scene = SceneType.Night;

        // HDR: simultaneously large dark AND bright regions + wide tonal spread.
        // Threshold lowered (68→60, 0.12→0.10) so borderline HDR landscapes
        // with lStdDev 60–68 are no longer misclassified as Portrait.
        else if (lStdDev > 60f && darkRatio > 0.10f && brightRatio > 0.10f)
            scene = SceneType.HDR;

        // Strong landscape: very dominant sky or green — checked BEFORE Portrait so
        // that blue-sky or lush landscapes are never misread as skin-tone photos.
        else if (skyRatio > 0.22f || greenRatio > 0.30f)
            scene = SceneType.Landscape;

        // Portrait: skin tones are the strongest cue.
        // Guards:
        //   - Landscape signals must be absent (sky, green)
        //   - Warm ratio must not dominate — red rock, sand, autumn foliage
        //     all have warmRatio > 0.08 without real skin
        //   - skinRatio raised 0.12→0.17: requires more skin coverage before
        //     committing, reducing false positives on earthy landscapes
        else if (skinRatio > 0.17f && skyRatio < 0.12f && greenRatio < 0.18f && warmRatio < 0.08f)
            scene = SceneType.Portrait;

        // Sunset / golden hour: dominant warm cast, mid-range brightness
        else if (warmRatio > 0.12f && lMean > 70 && lMean < 165)
            scene = SceneType.Sunset;

        // Landscape: sky-blue or green-vegetation dominant (moderate signals)
        else if (skyRatio > 0.15f || greenRatio > 0.20f)
            scene = SceneType.Landscape;

        // Low-light: dim but not black-night dark (candle, indoor evening, overcast)
        else if (lMean < 62 || darkRatio > 0.45f)
            scene = SceneType.LowLight;

        // High-key: very bright with minimal shadow (studio, snow, beach, whiteboard)
        else if (lMean > 162 && darkRatio < 0.05f)
            scene = SceneType.HighKey;

        // Daylight: well-lit scene with healthy tonal spread — the "good light" baseline
        else if (lMean > 95 && lStdDev > 28f)
            scene = SceneType.Daylight;

        else
            scene = SceneType.General;

        return new ImageAnalysis
        {
            LMean = lMean, Contrast = contrast, LStdDev = lStdDev,
            DarkRatio = darkRatio, BrightRatio = brightRatio,
            RMean = rMean, GMean = gMean, BMean = bMean,
            SkinRatio = skinRatio, SkyRatio = skyRatio,
            GreenRatio = greenRatio, WarmRatio = warmRatio,
            HotspotRatio = hotspotRatio,
            Scene = scene
        };
    }

    // ── Compute the "target" auto enhancement params ──
    // This is what slider = 0 (center) will apply.
    // Think of it as: "what would a smart editor do to make this photo look best?"
    public static AdjustmentState ComputeAutoParams(ImageAnalysis a)
    {
        var p = new AdjustmentState();

        float wb = a.RMean - a.BMean;
        float cs = Math.Max(a.RMean, Math.Max(a.GMean, a.BMean))
                 - Math.Min(a.RMean, Math.Min(a.GMean, a.BMean));

        switch (a.Scene)
        {
            // ──────────────────────────────────────────────────────────────
            case SceneType.Night:
            // Very dark, noisy, with bright hotspots. Goal: reveal the
            // ambient scene without blowing the light sources.
            // ──────────────────────────────────────────────────────────────
                p.Exposure   = Clamp((float)Math.Round((80f - a.LMean) * 0.55f), 0, 60);
                p.Brilliance = 20;
                p.Highlights = a.HotspotRatio > 0.04f ? -38 : -22; // protect blown lights
                p.Shadows    = a.DarkRatio > 0.70f ? 45 : 35;
                p.Contrast   = 22;
                p.Brightness = 5;
                p.BlackPoint = 8;  // crush blacks for deep inky shadows
                p.Saturation = cs < 10 ? 5 : 0;
                p.Vibrance   = 18;
                p.Warmth     = wb < -15 ? 10 : 0;  // city lights are warm; moonlight is cool
                p.Sharpness  = 8;  // careful — noise amplified by sharpening
                p.Definition = 10;
                p.Noise      = a.LMean < 30 ? 45 : 32; // heavy NR for night shots
                p.Vignette   = 16; // vignette draws eye to centre, deepens corners
                break;

            // ──────────────────────────────────────────────────────────────
            case SceneType.HDR:
            // Both bright and dark zones in the same frame. Goal: compress
            // the range into something displayable without losing local detail.
            // ──────────────────────────────────────────────────────────────
                p.Exposure   = Clamp((float)Math.Round((114f - a.LMean) * 0.30f), -30, 28);
                p.Brilliance = 15;
                p.Highlights = Clamp((float)Math.Round(-a.BrightRatio * 125f), -72, -25);
                p.Shadows    = Clamp((float)Math.Round(a.DarkRatio * 88f), 22, 58);
                p.Contrast   = -5; // reduce global contrast — HDR already has plenty
                p.Brightness = 0;
                p.BlackPoint = 12; // deepen shadow blacks for 3-D separation
                p.Saturation = cs < 20 ? 8 : 4;
                p.Vibrance   = cs < 20 ? 20 : 14;
                p.Warmth     = wb > 30 ? Clamp(-(float)Math.Round(wb * .25f), -20, -5)
                             : wb < -30 ? Clamp((float)Math.Round(-wb * .25f), 5, 20) : 0;
                p.Sharpness  = 10;
                p.Definition = 20; // local texture helps HDR look natural
                p.Noise      = a.LMean < 80 ? 15 : 5;
                p.Vignette   = 10;
                break;

            // ──────────────────────────────────────────────────────────────
            case SceneType.Portrait:
            // Skin-tone dominant. Goal: flattering brightness, soft contrast,
            // gentle saturation — keep skin looking natural.
            // ──────────────────────────────────────────────────────────────
                p.Exposure   = Clamp((float)Math.Round((122f - a.LMean) * 0.42f), -55, 50);
                p.Brilliance = a.LMean < 110 ? 20 : a.LMean < 140 ? 14 : 8;
                p.Highlights = a.BrightRatio > 0.10f ? -22 : -5;
                p.Shadows    = a.DarkRatio > 0.20f ? 32 : a.DarkRatio > 0.10f ? 20 : 10;
                p.Contrast   = a.Contrast < 30 ? 12 : a.Contrast < 50 ? 7 : 2;
                p.Brightness = a.LMean < 80 ? 5 : 0;
                p.BlackPoint = 8;  // subtle depth without flattening skin
                p.Saturation = cs < 20 ? 8  : cs < 40 ? 4 : 0;
                p.Vibrance   = cs < 20 ? 18 : cs < 40 ? 12 : 6;
                p.Warmth     = wb > 40 ? -8 : wb < -20 ? 15 : wb < 0 ? 8 : 3;
                p.Sharpness  = 8;   // soft sharpening — pores don't need crispness
                p.Definition = 12;
                p.Noise      = a.LMean < 70 ? 15 : 0;
                p.Vignette   = 10;  // draws focus to subject
                break;

            // ──────────────────────────────────────────────────────────────
            case SceneType.Sunset:
            // Warm golden tones. Goal: preserve the mood, deepen the golds,
            // protect sky detail without cooling the warmth.
            // ──────────────────────────────────────────────────────────────
                p.Exposure   = Clamp((float)Math.Round((100f - a.LMean) * 0.42f), -50, 36);
                p.Brilliance = 6;
                p.Highlights = a.BrightRatio > 0.15f
                             ? Clamp((float)Math.Round(-a.BrightRatio * 100f), -65, -22)
                             : -22;
                p.Shadows    = 15;
                p.Contrast   = a.Contrast < 35 ? 18 : 12;
                p.Brightness = 0;
                p.BlackPoint = 14; // deep shadows make golden tones pop
                p.Saturation = 22;
                p.Vibrance   = 32;
                p.Warmth     = wb > 20 ? 0 : 12; // never cool a sunset
                p.Sharpness  = 12;
                p.Definition = 16;
                p.Noise      = 0;
                p.Vignette   = 16; // dramatic vignette adds moodiness
                break;

            // ──────────────────────────────────────────────────────────────
            case SceneType.Landscape:
            // Sky/vegetation dominant. Goal: punchy contrast, vivid colour,
            // crisp details.
            // ──────────────────────────────────────────────────────────────
                p.Exposure   = Clamp((float)Math.Round((108f - a.LMean) * 0.42f), -55, 50);
                p.Brilliance = a.LMean < 110 ? 14 : 10;
                p.Highlights = a.BrightRatio > 0.15f
                             ? Clamp((float)Math.Round(-a.BrightRatio * 95f), -65, -8)
                             : a.BrightRatio > 0.06f ? -18 : -5;
                p.Shadows    = a.DarkRatio > 0.30f
                             ? Clamp((float)Math.Round(a.DarkRatio * 72f), 12, 46)
                             : a.DarkRatio > 0.12f ? 15 : 6;
                p.Contrast   = a.Contrast < 40 ? 24 : a.Contrast < 60 ? 14 : 7;
                p.Brightness = a.LMean < 80 ? 5 : 0;
                p.BlackPoint = 12; // richer shadow blacks ground the scene
                p.Saturation = cs < 20 ? 18 : cs < 40 ? 12 : 6;
                p.Vibrance   = cs < 20 ? 28 : cs < 40 ? 20 : 12;
                p.Warmth     = wb > 35 ? Clamp(-(float)Math.Round(wb * .28f), -28, -5)
                             : wb < -25 ? Clamp((float)Math.Round(-wb * .25f), 5, 22) : 0;
                p.Sharpness  = 14;
                p.Definition = 20;
                p.Noise      = 0;
                p.Vignette   = 12;
                break;

            // ──────────────────────────────────────────────────────────────
            case SceneType.LowLight:
            // Dim but not pitch black. Goal: lift the scene, add contrast
            // punch, keep noise under control.
            // ──────────────────────────────────────────────────────────────
                p.Exposure   = Clamp((float)Math.Round((95f - a.LMean) * 0.42f), -20, 52);
                p.Brilliance = 14;
                p.Highlights = a.BrightRatio > 0.06f ? -18 : -5;
                p.Shadows    = a.DarkRatio > 0.50f ? 26 : a.DarkRatio > 0.35f ? 18 : 12;
                p.Contrast   = a.Contrast < 35 ? 22 : a.Contrast < 55 ? 14 : 6;
                p.Brightness = a.LMean < 80 ? 5 : 0;
                p.BlackPoint = 8;  // deepen shadow floor without crushing legibility
                p.Saturation = cs < 15 ? 8 : 4;
                p.Vibrance   = cs < 15 ? 20 : 14;
                p.Warmth     = wb > 30 ? Clamp(-(float)Math.Round(wb * .3f), -25, -5)
                             : wb < -30 ? Clamp((float)Math.Round(-wb * .3f), 5, 25) : 0;
                p.Sharpness  = 10;
                p.Definition = 14;
                p.Noise      = a.DarkRatio > 0.50f ? 25 : 20;
                p.Vignette   = 8;
                break;

            // ──────────────────────────────────────────────────────────────
            case SceneType.HighKey:
            // Very bright, minimal shadows. Goal: protect highlights from
            // blowing, add gentle tonal separation.
            // ──────────────────────────────────────────────────────────────
                p.Exposure   = Clamp((float)Math.Round((140f - a.LMean) * 0.35f), -30, 18);
                p.Brilliance = 6;
                p.Highlights = Clamp((float)Math.Round(-a.BrightRatio * 85f), -52, -10);
                p.Shadows    = 4;
                p.Contrast   = a.Contrast < 25 ? 14 : 6;
                p.Brightness = 0;
                p.BlackPoint = 5;  // gentle tonal anchor
                p.Saturation = cs < 15 ? 8 : 4;
                p.Vibrance   = cs < 15 ? 14 : 8;
                p.Warmth     = wb > 30 ? -8 : wb < -30 ? 8 : 0;
                p.Sharpness  = 12;
                p.Definition = 14;
                p.Noise      = 0;
                p.Vignette   = 6;  // subtle, keeps high-key feel
                break;

            // ──────────────────────────────────────────────────────────────
            case SceneType.Daylight:
            // Well-lit scene, healthy tonal spread. Light corrections only —
            // the photo already has good light.
            // ──────────────────────────────────────────────────────────────
                p.Exposure   = Clamp((float)Math.Round((112f - a.LMean) * 0.30f), -25, 22);
                p.Brilliance = a.LMean < 115 ? 12 : 8;
                p.Highlights = a.BrightRatio > 0.12f ? -20 : a.BrightRatio > 0.06f ? -12 : -4;
                p.Shadows    = a.DarkRatio > 0.15f ? 12 : 5;
                p.Contrast   = a.Contrast < 40 ? 16 : a.Contrast < 58 ? 10 : 2;
                p.Brightness = 0;
                p.BlackPoint = 10; // adds dimension to otherwise flat daylight shots
                p.Saturation = cs < 20 ? 10 : cs < 40 ? 5 : 0;
                p.Vibrance   = cs < 20 ? 18 : cs < 40 ? 12 : 8;
                p.Warmth     = wb > 30 ? Clamp(-(float)Math.Round(wb * .25f), -22, -5)
                             : wb < -30 ? Clamp((float)Math.Round(-wb * .25f), 5, 22) : 0;
                p.Sharpness  = 12;
                p.Definition = 16;
                p.Noise      = 0;
                p.Vignette   = 10;
                break;

            // ──────────────────────────────────────────────────────────────
            default: // SceneType.General
            // ──────────────────────────────────────────────────────────────
                p.Exposure   = Clamp((float)Math.Round((112f - a.LMean) * 0.42f), -55, 50);
                p.Brilliance = a.LMean < 100 ? 16 : a.LMean < 130 ? 10 : 6;
                p.Highlights = a.BrightRatio > 0.15f
                             ? Clamp((float)Math.Round(-a.BrightRatio * 95f), -65, -8)
                             : a.BrightRatio > 0.06f ? -16 : -4;
                p.Shadows    = a.DarkRatio > 0.30f
                             ? Clamp((float)Math.Round(a.DarkRatio * 72f), 12, 46)
                             : a.DarkRatio > 0.12f ? 15 : 6;
                p.Contrast   = a.Contrast < 35
                             ? Clamp((float)Math.Round(20 + (35 - a.Contrast) * .4f), 10, 32)
                             : a.Contrast < 55 ? 10 : 2;
                p.Brightness = a.LMean < 80 ? 5 : 0;
                p.BlackPoint = 10; // ground the tonal range — adds perceived depth
                p.Saturation = cs < 15 ? 12 : cs < 35 ? 6 : 2;
                p.Vibrance   = cs < 15 ? 22 : cs < 35 ? 16 : 10;
                p.Warmth     = wb > 30 ? Clamp(-(float)Math.Round(wb * .3f), -28, -5)
                             : wb < -30 ? Clamp((float)Math.Round(-wb * .3f), 5, 28) : 0;
                p.Sharpness  = 12;
                p.Definition = 18;
                p.Noise      = a.LMean < 70 ? 12 : 0;
                p.Vignette   = 10;
                break;
        }

        // ── Clamp all values to slider ranges ──
        p.Exposure   = Math.Clamp(p.Exposure,   -100, 100);
        p.Brilliance = Math.Clamp(p.Brilliance, -100, 100);
        p.Highlights = Math.Clamp(p.Highlights, -100, 100);
        p.Shadows    = Math.Clamp(p.Shadows,    -100, 100);
        p.Contrast   = Math.Clamp(p.Contrast,   -100, 100);
        p.Brightness = Math.Clamp(p.Brightness, -100, 100);
        p.BlackPoint = Math.Clamp(p.BlackPoint, -100, 100);
        p.Saturation = Math.Clamp(p.Saturation, -100, 100);
        p.Vibrance   = Math.Clamp(p.Vibrance,   -100, 100);
        p.Warmth     = Math.Clamp(p.Warmth,     -100, 100);
        p.Sharpness  = Math.Clamp(p.Sharpness,  0,    100);
        p.Definition = Math.Clamp(p.Definition, 0,    100);
        p.Noise      = Math.Clamp(p.Noise,      0,    100);
        p.Vignette   = Math.Clamp(p.Vignette,   0,    100);

        return p;
    }

    // ── Apply filter with intensity blend ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float r, float g, float b) ApplyFilter(
        float r, float g, float b, int x, int y, FilterType filter, float intensity)
    {
        float t = intensity / 100f;
        (float nr, float ng, float nb) = filter switch
        {
            FilterType.Vivid      => (r * 1.07f, g * 1.03f, b * 1.09f),
            FilterType.Warm       => (r * 1.1f + 8, g * 1.01f, b * .9f - 5),
            FilterType.Cool       => (r * .93f - 5, g * 1.01f, b * 1.12f + 8),
            FilterType.Sunset     => (r * 1.15f + 18, g * 0.96f + 4, b * 0.75f - 12),
            FilterType.TealOrange => (r * 1.12f + 10, g * 0.97f, b * 0.80f - 8),
            FilterType.Process    => (r * 1.01f, g * 1.09f + 5, b * .97f + 2),
            FilterType.Lush       => (r * 0.92f - 4, g * 1.12f + 8, b * 0.94f - 2),
            FilterType.Retro      => (r * 0.95f + 12, g * 1.06f + 8, b * 0.80f + 15),
            FilterType.Instant    => (r * 0.90f + 32, g * 0.88f + 22, b * 0.80f + 28),
            FilterType.Fade       => (r * .87f + 26, g * .87f + 23, b * .87f + 20),
            FilterType.ZoomBlur   => (r, g, b),   // handled as post-process in Render()
            FilterType.Grainy     => GrainyPixel(r, g, b, x, y),
            FilterType.Gritty     => GrittyPixel(r, g, b, x, y),
            FilterType.Chrome     => ((r - 10) * 1.22f, (g - 6) * 1.15f + 2, (b - 4) * 1.10f + 6),
            FilterType.Sepia      => (r * .299f + g * .587f + b * .114f) is float lm ?
                                     (lm * 1.08f + 20, lm * 0.94f + 5, lm * 0.75f - 10) :
                                     (r, g, b),
            FilterType.Silvertone => (r * .299f + g * .587f + b * .114f) is float ys ?
                                     (ys * 1.07f + 8, ys * 1.03f + 4, ys * .95f - 4) :
                                     (r, g, b),
            FilterType.Matte      => (r * .299f + g * .587f + b * .114f) is float ml ?
                                     (ml + (r - ml) * 0.72f + 24,
                                      ml + (g - ml) * 0.72f + 22,
                                      ml + (b - ml) * 0.72f + 20) :
                                     (r, g, b),
            FilterType.Dramatic   => ((r - 20) * 1.18f, (g - 20) * 1.18f, (b - 20) * 1.18f),
            FilterType.Mono       => (r * .299f + g * .587f + b * .114f,
                                      r * .299f + g * .587f + b * .114f,
                                      r * .299f + g * .587f + b * .114f),
            FilterType.Noir       => ((r * .299f + g * .587f + b * .114f - 20) * 1.35f,
                                      (r * .299f + g * .587f + b * .114f - 20) * 1.35f,
                                      (r * .299f + g * .587f + b * .114f - 20) * 1.35f),
            _                     => (r, g, b)
        };
        return (r + (nr - r) * t, g + (ng - g) * t, b + (nb - b) * t);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float r, float g, float b) GrainyPixel(float r, float g, float b, int x, int y)
    {
        // Deterministic film grain — fast integer hash per pixel
        uint h1 = (uint)(x * 374761393 + y * 1301081); h1 ^= h1 >> 13; h1 *= 1274126177u; h1 ^= h1 >> 16;
        uint h2 = (uint)(x * 1557985979 + y * 1012321331); h2 ^= h2 >> 13; h2 *= 2246822519u; h2 ^= h2 >> 16;
        float luma = (h1 & 0xFF) / 255f * 2f - 1f;   // -1 to +1
        float chroma = (h2 & 0xFF) / 255f * 2f - 1f; // subtle color shift
        float grain = luma * 42f;      // ±42 luma grain
        float color = chroma * 10f;    // ±10 color grain
        return (r + grain + color, g + grain, b + grain - color);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float r, float g, float b) GrittyPixel(float r, float g, float b, int x, int y)
    {
        // Desaturate partially
        float lm = r * .299f + g * .587f + b * .114f;
        float gr = lm + (r - lm) * 0.50f;
        float gg = lm + (g - lm) * 0.50f;
        float gb = lm + (b - lm) * 0.50f;
        // Boost contrast
        gr = (gr - 128) * 1.30f + 128;
        gg = (gg - 128) * 1.30f + 128;
        gb = (gb - 128) * 1.30f + 128;
        // Crush blacks
        float bl = 18f, scl = 255f / (255f - bl);
        gr = (gr - bl) * scl; gg = (gg - bl) * scl; gb = (gb - bl) * scl;
        // Subtle warm grime cast
        gr += 6f; gb -= 4f;
        // Film grain
        uint hg = (uint)(x * 374761393 ^ y * 1301081); hg ^= hg >> 13; hg *= 1274126177u; hg ^= hg >> 16;
        float grit = ((hg & 0xFF) / 255f * 2f - 1f) * 22f;
        return (gr + grit, gg + grit * 0.85f, gb + grit * 0.75f);
    }

    // ── Radial zoom-blur — applied as post-process after the main pixel loop ──
    private static void ApplyZoomBlur(byte[] buf, int w, int h, float intensity)
    {
        const int samples = 10;
        float cx = w * 0.5f, cy = h * 0.5f;
        float maxPull = 0.05f + (intensity / 100f) * 0.15f; // 5–20% radial pull
        var src = (byte[])buf.Clone();
        int stride = w * 4;

        System.Threading.Tasks.Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                float dx = cx - x, dy = cy - y;
                float dist = MathF.Sqrt(dx * dx + dy * dy);
                if (dist < 2f) continue;

                float blurLen = dist * maxPull;
                float nx = dx / dist, ny = dy / dist; // direction to centre

                float rb = 0, gb = 0, bb = 0;
                for (int s = 0; s < samples; s++)
                {
                    float t = (float)s / (samples - 1);
                    int ix = (int)Math.Clamp(x + nx * blurLen * t, 0, w - 1);
                    int iy = (int)Math.Clamp(y + ny * blurLen * t, 0, h - 1);
                    int idx = iy * stride + ix * 4;
                    rb += src[idx]; gb += src[idx + 1]; bb += src[idx + 2];
                }
                int di = y * stride + x * 4;
                buf[di]     = (byte)(rb / samples);
                buf[di + 1] = (byte)(gb / samples);
                buf[di + 2] = (byte)(bb / samples);
                // alpha [di+3] unchanged
            }
        });
    }

    // ── RGB ↔ HSL conversion helpers ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float h, float s, float l) RgbToHsl(float r, float g, float b)
    {
        r /= 255f; g /= 255f; b /= 255f;
        float max = MathF.Max(r, MathF.Max(g, b));
        float min = MathF.Min(r, MathF.Min(g, b));
        float l = (max + min) * 0.5f;
        if (max == min) return (0f, 0f, l);
        float d = max - min;
        float s = l > 0.5f ? d / (2f - max - min) : d / (max + min);
        float h;
        if      (max == r) h = ((g - b) / d + (g < b ? 6f : 0f)) / 6f;
        else if (max == g) h = ((b - r) / d + 2f) / 6f;
        else               h = ((r - g) / d + 4f) / 6f;
        return (h, s, l);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HueToRgb(float p, float q, float t)
    {
        if (t < 0) t += 1f;
        if (t > 1) t -= 1f;
        if (t < 1/6f) return p + (q - p) * 6f * t;
        if (t < 0.5f) return q;
        if (t < 2/3f) return p + (q - p) * (2/3f - t) * 6f;
        return p;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float r, float g, float b) HslToRgb(float h, float s, float l)
    {
        if (s == 0f) { float v = l * 255f; return (v, v, v); }
        float q = l < 0.5f ? l * (1f + s) : l + s - l * s;
        float p = 2f * l - q;
        return (HueToRgb(p, q, h + 1/3f) * 255f,
                HueToRgb(p, q, h)         * 255f,
                HueToRgb(p, q, h - 1/3f) * 255f);
    }

    // Hue channel centers (degrees): Red=0, Orange=30, Yellow=60, Green=120, Cyan=190, Blue=240, Purple=300
    private static readonly float[] HslCenters = { 0f, 30f, 60f, 120f, 190f, 240f, 300f };
    private static readonly float[] HslWidths  = { 40f, 35f, 40f, 70f,  40f,  70f,  60f };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HslChannelWeight(float hueDeg, int ch)
    {
        float d = MathF.Abs(hueDeg - HslCenters[ch]);
        if (d > 180f) d = 360f - d;
        float half = HslWidths[ch] * 0.5f;
        return d < half ? 1f - d / half : 0f;
    }

    private static (float hShift, float sSat) GetHslAdj(AdjustmentState s, int ch) => ch switch
    {
        0 => (s.HslRH, s.HslRS),
        1 => (s.HslOH, s.HslOS),
        2 => (s.HslYH, s.HslYS),
        3 => (s.HslGH, s.HslGS),
        4 => (s.HslCH, s.HslCS),
        5 => (s.HslBH, s.HslBS),
        _ => (s.HslPH, s.HslPS),
    };

    // ── Tone-curve LUT — returns null when curve is identity ──
    public static byte[]? ComputeCurveLUT(AdjustmentState s)
    {
        if (!s.HasCurve) return null;
        float[] xs = { 0f, 64f, 128f, 192f, 255f };
        float[] ys = { s.Curve0, s.Curve64, s.Curve128, s.Curve192, s.Curve255 };
        var lut = new byte[256];
        for (int i = 0; i < 256; i++)
            lut[i] = (byte)Math.Clamp(CatmullRom(xs, ys, i), 0f, 255f);
        return lut;
    }

    private static float CatmullRom(float[] xs, float[] ys, float x)
    {
        int n = xs.Length;
        int seg = n - 2;
        for (int i = 0; i < n - 1; i++) if (x <= xs[i + 1]) { seg = i; break; }
        int i0 = Math.Max(0, seg - 1), i1 = seg,
            i2 = Math.Min(n - 1, seg + 1), i3 = Math.Min(n - 1, seg + 2);
        float span = xs[i2] - xs[i1];
        float t = span > 0 ? (x - xs[i1]) / span : 0f;
        float t2 = t * t, t3 = t2 * t;
        float p0 = ys[i0], p1 = ys[i1], p2 = ys[i2], p3 = ys[i3];
        return 0.5f * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t2 + (-p0+3*p1-3*p2+p3)*t3);
    }

    // ── Histogram — returns [256]R, [256]G, [256]B, [256]L counts ──
    public static (int[] r, int[] g, int[] b, int[] lum) ComputeHistogram(byte[] pixels, int w, int h)
    {
        var rH = new int[256]; var gH = new int[256];
        var bH = new int[256]; var lH = new int[256];
        int step = Math.Max(1, (int)Math.Sqrt((double)w * h / 60000.0));
        for (int y = 0; y < h; y += step)
        for (int x = 0; x < w;  x += step)
        {
            int i = (y * w + x) * 4;
            byte bv = pixels[i], gv = pixels[i+1], rv = pixels[i+2];
            rH[rv]++; gH[gv]++; bH[bv]++;
            lH[(byte)(rv * .299f + gv * .587f + bv * .114f)]++;
        }
        return (rH, gH, bH, lH);
    }

    // ── Per-pixel adjustment ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float r, float g, float b) AdjustPixel(
        float r, float g, float b, int x, int y, int w, int h, AdjustmentState s,
        byte[]? curveLut)
    {
        if (s.Filter != FilterType.None)
            (r, g, b) = ApplyFilter(r, g, b, x, y, s.Filter, s.FilterIntensity);

        float exp = MathF.Pow(2f, s.Exposure / 100f);
        r *= exp; g *= exp; b *= exp;

        float br = s.Brightness * 1.4f;
        r += br; g += br; b += br;

        float co = 1f + s.Contrast / 100f;
        r = (r - 128) * co + 128;
        g = (g - 128) * co + 128;
        b = (b - 128) * co + 128;

        float lum = r * .299f + g * .587f + b * .114f;
        float hi  = lum > 128 ? (lum - 128) / 127f : 0f;
        float sh  = lum < 128 ? (128 - lum) / 128f : 0f;
        float hiD = s.Highlights * -.9f * hi;
        float shD = s.Shadows    *  1.1f * sh;
        r += hiD + shD; g += hiD + shD; b += hiD + shD;

        float midPush = (128 - lum) * (s.Brilliance / 100f) * .36f;
        r += midPush; g += midPush; b += midPush;

        if (s.BlackPoint > 0)
        {
            // Crush blacks: lift the black point, making darks darker
            float bl = s.BlackPoint * 1.25f;
            float scale = 255f / Math.Max(1, 255f - bl);
            r = (r - bl) * scale;
            g = (g - bl) * scale;
            b = (b - bl) * scale;
        }
        else if (s.BlackPoint < 0)
        {
            // Lift blacks: raise shadow floor (milky/fade effect)
            float lift = -s.BlackPoint * 0.6f;
            r = r + lift * (1f - r / 255f);
            g = g + lift * (1f - g / 255f);
            b = b + lift * (1f - b / 255f);
        }

        lum = r * .299f + g * .587f + b * .114f;
        float sat = 1f + s.Saturation / 100f;
        r = lum + (r - lum) * sat;
        g = lum + (g - lum) * sat;
        b = lum + (b - lum) * sat;

        if (s.Vibrance != 0)
        {
            float vib  = s.Vibrance / 100f;
            float maxc = Math.Max(r, Math.Max(g, b));
            float avg  = (r + g + b) / 3f;
            float boost = 1f + vib * (1f - Math.Abs(maxc - avg) / 128f);
            r = avg + (r - avg) * boost;
            g = avg + (g - avg) * boost;
            b = avg + (b - avg) * boost;
        }

        r += s.Warmth * .55f; b -= s.Warmth * .55f;
        r += s.Tint * .28f;   b += s.Tint * .28f; g -= s.Tint * .18f;

        if (s.Vignette > 0)
        {
            float dx = (x - w / 2f) / (w / 2f);
            float dy = (y - h / 2f) / (h / 2f);
            float dist = MathF.Sqrt(dx * dx + dy * dy);
            float vig = 1f - Math.Max(0, dist - .35f) * (s.Vignette / 85f);
            r *= vig; g *= vig; b *= vig;
        }

        // ── Tone curve ──
        if (curveLut != null)
        {
            r = curveLut[(byte)Clamp(r)];
            g = curveLut[(byte)Clamp(g)];
            b = curveLut[(byte)Clamp(b)];
        }

        // ── HSL Mixer ──
        if (s.HasHsl)
        {
            var (hue, hs, hl) = RgbToHsl(Clamp(r), Clamp(g), Clamp(b));
            if (hs > 0.04f)
            {
                float hueDeg = hue * 360f;
                float totalHShift = 0f, totalSBoost = 0f, totalW = 0f;
                for (int ch = 0; ch < 7; ch++)
                {
                    float w2 = HslChannelWeight(hueDeg, ch);
                    if (w2 <= 0f) continue;
                    var (hShift, sSat) = GetHslAdj(s, ch);
                    totalHShift += hShift * w2;
                    totalSBoost += sSat  * w2;
                    totalW      += w2;
                }
                if (totalW > 0f)
                {
                    hue  = (hue + totalHShift / totalW / 180f + 1f) % 1f;
                    hs = Math.Clamp(hs + totalSBoost / totalW / 100f, 0f, 1f);
                    (r, g, b) = HslToRgb(hue, hs, hl);
                }
            }
        }

        return (Clamp(r), Clamp(g), Clamp(b));
    }

    // ── Inverse geometric transform: maps output (x,y) → source (xSrc,ySrc) ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float xSrc, float ySrc) InverseTransform(float x, float y, int w, int h, AdjustmentState s)
    {
        float cx = w * 0.5f, cy = h * 0.5f;
        float nx = x - cx, ny = y - cy;

        // Z-axis rotation with auto-fit: rotated image is scaled to always fill
        // the display canvas without clipping — black corners appear naturally.
        if (s.RotateZ != 0f)
        {
            float theta = s.RotateZ * MathF.PI / 180f;
            float absC = MathF.Abs(MathF.Cos(theta));
            float absS = MathF.Abs(MathF.Sin(theta));
            // Bounding box of rotated image; fitScale shrinks display so it fits
            float fitScale = MathF.Max(
                (absC * w + absS * h) / w,
                (absS * w + absC * h) / h);
            // Inverse rotation applied to fitScale-expanded coordinates
            float rad = -theta;
            float cosR = MathF.Cos(rad), sinR = MathF.Sin(rad);
            float snx = nx * fitScale, sny = ny * fitScale;
            nx = cosR * snx - sinR * sny;
            ny = sinR * snx + cosR * sny;
        }

        // Y-axis horizontal keystone (perspective left/right tilt)
        if (s.TiltY != 0f)
        {
            float t = s.TiltY * 0.003f;          // softer: max ±0.3 at ±100
            float yNorm = ny / (cy < 1f ? 1f : cy);
            float scaleX = 1f + t * yNorm;
            scaleX = MathF.Max(0.05f, scaleX);   // prevent inversion
            nx /= scaleX;
        }

        // X-axis vertical keystone (perspective up/down tilt)
        if (s.TiltX != 0f)
        {
            float t = s.TiltX * 0.003f;
            float xNorm = nx / (cx < 1f ? 1f : cx);
            float scaleY = 1f + t * xNorm;
            scaleY = MathF.Max(0.05f, scaleY);
            ny /= scaleY;
        }

        return (nx + cx, ny + cy);
    }

    // ── Bilinear sample — out-of-bounds returns transparent black ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float r, float g, float b, byte a) BilinearSample(byte[] pixels, int w, int h, float xf, float yf)
    {
        int x0 = (int)xf, y0 = (int)yf;
        if (x0 < 0 || y0 < 0 || x0 >= w || y0 >= h) return (0f, 0f, 0f, 0);
        float fx = xf - x0, fy = yf - y0;
        int x1 = Math.Min(x0 + 1, w - 1), y1 = Math.Min(y0 + 1, h - 1);
        int i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4;
        int i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
        float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
        float w01 = (1 - fx) * fy,       w11 = fx * fy;
        float b = pixels[i00]   * w00 + pixels[i10]   * w10 + pixels[i01]   * w01 + pixels[i11]   * w11;
        float g = pixels[i00+1] * w00 + pixels[i10+1] * w10 + pixels[i01+1] * w01 + pixels[i11+1] * w11;
        float r = pixels[i00+2] * w00 + pixels[i10+2] * w10 + pixels[i01+2] * w01 + pixels[i11+2] * w11;
        byte  a = (byte)(pixels[i00+3] * w00 + pixels[i10+3] * w10 + pixels[i01+3] * w01 + pixels[i11+3] * w11);
        return (r, g, b, a);
    }

    // ── Render to raw pixel buffer ──
    public static byte[] RenderToBuffer(byte[] sourcePixels, int width, int height, AdjustmentState adj)
    {
        int stride = width * 4;
        byte[] buf = new byte[height * stride];
        bool hasTransform = adj.RotateZ != 0f || adj.TiltX != 0f || adj.TiltY != 0f;
        byte[]? curveLut = ComputeCurveLUT(adj);

        if (width * height > 500_000)
        {
            System.Threading.Tasks.Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    float r, g, b; byte a;
                    if (hasTransform)
                    {
                        var (xSrc, ySrc) = InverseTransform(x, y, width, height, adj);
                        (r, g, b, a) = BilinearSample(sourcePixels, width, height, xSrc, ySrc);
                    }
                    else
                    {
                        int si = (y * width + x) * 4;
                        b = sourcePixels[si]; g = sourcePixels[si + 1];
                        r = sourcePixels[si + 2]; a = sourcePixels[si + 3];
                    }
                    (r, g, b) = AdjustPixel(r, g, b, x, y, width, height, adj, curveLut);
                    int di = y * stride + x * 4;
                    buf[di] = (byte)b; buf[di + 1] = (byte)g; buf[di + 2] = (byte)r; buf[di + 3] = a;
                }
            });
        }
        else
        {
            for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                float r, g, b; byte a;
                if (hasTransform)
                {
                    var (xSrc, ySrc) = InverseTransform(x, y, width, height, adj);
                    (r, g, b, a) = BilinearSample(sourcePixels, width, height, xSrc, ySrc);
                }
                else
                {
                    int si = (y * width + x) * 4;
                    b = sourcePixels[si]; g = sourcePixels[si + 1];
                    r = sourcePixels[si + 2]; a = sourcePixels[si + 3];
                }
                (r, g, b) = AdjustPixel(r, g, b, x, y, width, height, adj, curveLut);
                int di = y * stride + x * 4;
                buf[di] = (byte)b; buf[di + 1] = (byte)g; buf[di + 2] = (byte)r; buf[di + 3] = a;
            }
        }

        float sharpAmount = Math.Max(adj.Sharpness, adj.Definition) / 100f;
        if (sharpAmount > 0)
            Sharpen(buf, width, height, stride, sharpAmount);

        if (adj.Noise > 0)
            ReduceNoise(buf, width, height, stride, adj.Noise / 100f);

        if (adj.Filter == FilterType.ZoomBlur)
            ApplyZoomBlur(buf, width, height, adj.FilterIntensity);

        return buf;
    }

    public static WriteableBitmap BufferToBitmap(byte[] buf, int width, int height)
    {
        int stride = width * 4;
        var wb = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
        wb.WritePixels(new System.Windows.Int32Rect(0, 0, width, height), buf, stride, 0);
        wb.Freeze();
        return wb;
    }

    // ── Main render — parallel for large images ──
    public static WriteableBitmap Render(byte[] sourcePixels, int width, int height, AdjustmentState adj)
        => BufferToBitmap(RenderToBuffer(sourcePixels, width, height, adj), width, height);

    private static void Sharpen(byte[] buf, int w, int h, int stride, float amount)
    {
        byte[] src = (byte[])buf.Clone();
        float ctr  = 1f + amount * 4f;
        float side = -amount;
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            int i  = y * stride + x * 4;
            int up = Math.Max(0,   y - 1) * stride + x * 4;
            int dn = Math.Min(h-1, y + 1) * stride + x * 4;
            int lt = y * stride + Math.Max(0,   x - 1) * 4;
            int rt = y * stride + Math.Min(w-1, x + 1) * 4;
            for (int c = 0; c < 3; c++)
                buf[i + c] = (byte)Clamp(src[i+c]*ctr + (src[up+c]+src[dn+c]+src[lt+c]+src[rt+c])*side);
        }
    }

    private static void ReduceNoise(byte[] buf, int w, int h, int stride, float amount)
    {
        byte[] src = (byte[])buf.Clone();
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            int i = y * stride + x * 4;
            for (int c = 0; c < 3; c++)
            {
                float sum = 0; int cnt = 0;
                for (int oy = -1; oy <= 1; oy++)
                for (int ox = -1; ox <= 1; ox++)
                {
                    int xx = Math.Clamp(x + ox, 0, w - 1);
                    int yy = Math.Clamp(y + oy, 0, h - 1);
                    sum += src[yy * stride + xx * 4 + c];
                    cnt++;
                }
                buf[i + c] = (byte)(src[i + c] * (1 - amount) + (sum / cnt) * amount);
            }
        }
    }

    public static (byte[] pixels, int width, int height) LoadImageFile(string path)
    {
        var bi = new BitmapImage();
        bi.BeginInit();
        bi.UriSource = new Uri(path);
        bi.CacheOption = BitmapCacheOption.OnLoad;
        bi.EndInit();
        bi.Freeze();
        var conv = new FormatConvertedBitmap(bi, PixelFormats.Bgra32, null, 0);
        conv.Freeze();
        int w = conv.PixelWidth, h = conv.PixelHeight;
        int stride = w * 4;
        byte[] pixels = new byte[h * stride];
        conv.CopyPixels(pixels, stride, 0);
        return (pixels, w, h);
    }

    public static (byte[] pixels, int w, int h) Rotate(byte[] src, int w, int h, int degrees)
    {
        bool swap = degrees == 90 || degrees == -90 || degrees == 270;
        int nw = swap ? h : w, nh = swap ? w : h;
        byte[] dst = new byte[nw * nh * 4];
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            int si = (y * w + x) * 4;
            int dx, dy;
            if      (degrees ==  90) { dx = h - 1 - y; dy = x; }
            else if (degrees == -90 || degrees == 270) { dx = y; dy = w - 1 - x; }
            else    { dx = w - 1 - x; dy = h - 1 - y; }
            int di = (dy * nw + dx) * 4;
            Buffer.BlockCopy(src, si, dst, di, 4);
        }
        return (dst, nw, nh);
    }

    public static byte[] Flip(byte[] src, int w, int h, bool horizontal)
    {
        byte[] dst = new byte[src.Length];
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            int si = (y * w + x) * 4;
            int dx = horizontal ? w - 1 - x : x;
            int dy = horizontal ? y : h - 1 - y;
            int di = (dy * w + dx) * 4;
            Buffer.BlockCopy(src, si, dst, di, 4);
        }
        return dst;
    }

    public static (byte[] pixels, int w, int h) Crop(byte[] src, int srcW, int srcH,
        int cx, int cy, int cw, int ch)
    {
        cw = Math.Clamp(cw, 1, srcW - cx);
        ch = Math.Clamp(ch, 1, srcH - cy);
        byte[] dst = new byte[cw * ch * 4];
        for (int y = 0; y < ch; y++)
        {
            int si = ((cy + y) * srcW + cx) * 4;
            int di = y * cw * 4;
            Buffer.BlockCopy(src, si, dst, di, cw * 4);
        }
        return (dst, cw, ch);
    }

    // ── Neural-enhanced auto params ───────────────────────────────────────────

    /// <summary>
    /// Blend rule-based auto params with a correction informed by Places365-MobileNet
    /// scene weights. The NN can shift the result by at most 50% toward the params that
    /// would apply to the NN's dominant scene, weighted by confidence.
    /// Returns <paramref name="ruleParams"/> unchanged when NN agrees with the rule-based
    /// scene or when NN confidence is below the 0.25 threshold.
    /// </summary>
    public static AdjustmentState RefineWithNN(
        AdjustmentState ruleParams,
        ImageAnalysis   analysis,
        SceneWeights    weights)
    {
        SceneType nnScene = weights.Dominant;
        float     nnConf  = weights.DominantScore;

        if (nnScene == analysis.Scene || nnConf < 0.25f) return ruleParams;

        // Compute what ComputeAutoParams would produce for the NN's scene
        var nnAnalysis = new ImageAnalysis
        {
            LMean        = analysis.LMean,        Contrast    = analysis.Contrast,
            LStdDev      = analysis.LStdDev,      DarkRatio   = analysis.DarkRatio,
            BrightRatio  = analysis.BrightRatio,  RMean       = analysis.RMean,
            GMean        = analysis.GMean,         BMean       = analysis.BMean,
            SkinRatio    = analysis.SkinRatio,     SkyRatio    = analysis.SkyRatio,
            GreenRatio   = analysis.GreenRatio,    WarmRatio   = analysis.WarmRatio,
            HotspotRatio = analysis.HotspotRatio,
            Scene        = nnScene,
        };
        var nnParams = ComputeAutoParams(nnAnalysis);

        // Cap NN influence at 50% so rule-based always has majority weight
        float blend = Math.Min(nnConf * 0.6f, 0.5f);
        return LerpParams(ruleParams, nnParams, blend);
    }

    private static AdjustmentState LerpParams(AdjustmentState a, AdjustmentState b, float t)
    {
        static float L(float x, float y, float f) => x + (y - x) * f;
        return new AdjustmentState
        {
            Exposure   = L(a.Exposure,   b.Exposure,   t),
            Brilliance = L(a.Brilliance, b.Brilliance, t),
            Highlights = L(a.Highlights, b.Highlights, t),
            Shadows    = L(a.Shadows,    b.Shadows,    t),
            Contrast   = L(a.Contrast,   b.Contrast,   t),
            Brightness = L(a.Brightness, b.Brightness, t),
            BlackPoint = L(a.BlackPoint, b.BlackPoint, t),
            Saturation = L(a.Saturation, b.Saturation, t),
            Vibrance   = L(a.Vibrance,   b.Vibrance,   t),
            Warmth     = L(a.Warmth,     b.Warmth,     t),
            Tint       = L(a.Tint,       b.Tint,       t),
            Sharpness  = L(a.Sharpness,  b.Sharpness,  t),
            Definition = L(a.Definition, b.Definition, t),
            Noise      = L(a.Noise,      b.Noise,      t),
            Vignette   = L(a.Vignette,   b.Vignette,   t),
            Filter          = a.Filter,
            FilterIntensity = a.FilterIntensity,
        };
    }
}
