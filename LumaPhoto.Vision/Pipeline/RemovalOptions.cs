namespace LumaPhoto.Vision.Pipeline;

/// <summary>
/// Edge-quality tunables. This is the part that separates a $30 tool from a
/// GitHub demo — the raw network mask is always mushy at hair and fur, and
/// these controls are what clean it up. Expose them in the UI.
/// </summary>
public sealed record RemovalOptions
{
    /// <summary>
    /// Levels adjustment on the mask. Alpha below <see cref="AlphaFloor"/> snaps to
    /// fully transparent, above <see cref="AlphaCeiling"/> snaps to fully opaque, and
    /// the range between is stretched linearly. Kills the grey halo that upsampling
    /// leaves behind. Widen the gap to preserve wispy hair, narrow it for hard edges.
    /// </summary>
    public byte AlphaFloor { get; init; } = 20;

    public byte AlphaCeiling { get; init; } = 235;

    /// <summary>
    /// Box-blur radius in pixels applied to the mask, in source resolution.
    /// 0 disables. Softens the staircase from upsampling a 320px mask to 4000px.
    /// Scale this with image size — see <see cref="ForImageSize"/>.
    /// </summary>
    public int FeatherRadius { get; init; } = 2;

    /// <summary>
    /// S-curve strength on mask midtones, 0..1. Pulls semi-transparent pixels
    /// toward whichever side they're closer to, sharpening the transition
    /// without the jaggedness of a hard threshold.
    /// </summary>
    public float EdgeContrast { get; init; } = 0.35f;

    /// <summary>
    /// Shrinks the mask inward by this many pixels before feathering. A pixel or
    /// two removes the background colour fringe that survives on light subjects
    /// against dark backdrops. Set 0 for thin subjects that would erode away.
    /// </summary>
    public int Shrink { get; init; } = 1;

    public static RemovalOptions Default => new();

    /// <summary>Softer edges for hair, fur, and semi-transparent fabric.</summary>
    public static RemovalOptions SoftSubject => new()
    {
        AlphaFloor = 8,
        AlphaCeiling = 250,
        FeatherRadius = 3,
        EdgeContrast = 0.15f,
        Shrink = 0,
    };

    /// <summary>Crisp edges for product shots, logos, and packaging.</summary>
    public static RemovalOptions HardSubject => new()
    {
        AlphaFloor = 40,
        AlphaCeiling = 215,
        FeatherRadius = 1,
        EdgeContrast = 0.6f,
        Shrink = 1,
    };

    /// <summary>
    /// Scales feather and shrink with resolution. A 2px feather that looks right
    /// on a 1000px image is invisible on a 6000px one.
    /// </summary>
    public RemovalOptions ForImageSize(int width, int height)
    {
        float factor = Math.Max(width, height) / 1500f;
        if (factor <= 1f) return this;

        return this with
        {
            FeatherRadius = (int)Math.Round(FeatherRadius * factor),
            Shrink = (int)Math.Round(Shrink * factor),
        };
    }
}
