<?xml version='1.0'?>
<workbook>
  <preferences>
    <!-- Binary income encoding: <=50K blue, >50K orange.
         CVD-validated (protan/deutan/tritan dE > 96, contrast >= 3:1 on light surface). -->
    <color-palette name="Income CVD-safe" type="regular">
      <color>#2a78d6</color>
      <color>#eb6834</color>
    </color-palette>
    <!-- General categorical palette, fixed slot order chosen to maximise
         minimum adjacent CVD distance (used for workclass / QA series). -->
    <color-palette name="VA Categorical" type="regular">
      <color>#2a78d6</color>
      <color>#1baf7a</color>
      <color>#eda100</color>
      <color>#008300</color>
      <color>#4a3aa7</color>
      <color>#e34948</color>
      <color>#e87ba4</color>
      <color>#eb6834</color>
    </color-palette>
    <!-- Sequential single-hue blue ramp for quantitative colour
         (e.g. imputation uncertainty / entropy if surfaced). -->
    <color-palette name="VA Sequential Blue" type="ordered-sequential">
      <color>#cde2fb</color>
      <color>#9ec5f4</color>
      <color>#6da7ec</color>
      <color>#3987e5</color>
      <color>#256abf</color>
      <color>#184f95</color>
      <color>#0d366b</color>
    </color-palette>
  </preferences>
</workbook>
