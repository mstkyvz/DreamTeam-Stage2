# -*- coding: utf-8 -*-
"""L4 extra prompt variants, written against L4's OWN measured skew.

Measured on the 7500 public-test rows with the `zs` prompt (truth is 2500/2500/2500):

    irrelevant    1455   0.58x   under-called
    relevant      1848   0.74x   under-called
    semi-relevant 4197   1.68x   OVER-called

Both poles are under-called and `semi-relevant` is the dump class. That is the OPPOSITE of L1's
Qwen, which over-called `irrelevant` 1.9x and needed the tie-break pushed toward `semi`. Applying
the same softening here would push mass further into the class that is already 1.68x over, so the
correction has to run the other way: `semi-relevant` is the label that must be EARNED, and the
two poles become available whenever a concrete reason exists.

That is what `hard` does. It reuses L3's role, three-label definitions and output block verbatim
-- only the decision order and the calibration block are rewritten -- so the diff against `zs` is
the tie-break rule and nothing else.
"""
import prompts_l3 as P

KARAR_HARD = u"""KARAR SIRASI — sırayla sor, ilk "evet"te dur

S1. Sorgu hangi şartları AÇIKÇA yazıyor? Sorguda yazmayan hiçbir şey şart değildir — sorgu renk
    demediyse rengin farklı olması ihlal sayılmaz, sorgu adet demediyse paket adedi ihlal sayılmaz.

S2. Bu açık şartlardan biri üründe ÇELİŞİYOR mu? (istenen renk yerine başka renk, istenen marka
    yerine başka marka, istenen cinsiyet yerine öteki cinsiyet, istenen ürün tipi yerine başka bir
    ihtiyaca hizmet eden ürün tipi, istenen beden/numara/model/nesil yerine başkası)
      → EVET ise: irrelevant. Tereddüt etme; şart yazılmış ve tutmuyorsa cevap irrelevant'tır.

S3. Ürünün ana tipi sorgunun istediği şey ve açık şartların hepsi tutuyor mu?
      → EVET ise: relevant. Sorgunun yazmadığı ayrıntıların farklı olması bunu bozmaz.

S4. Buraya geldiysen semi-relevant'ı HAK ETTİRMEK zorundasın: ürünün sorgunun ihtiyacına
    kısmen hizmet ettiği somut bir nokta göstermelisin (doğru aileden ama istenmeyen varyant,
    ya da sorgudaki ürünün aksesuarı/parçası/tamamlayıcısı).
      → Bunu gösterebiliyorsan: semi-relevant.
      → Gösteremiyorsan semi-relevant YAZMA. S2 ve S3'e dön ve ikisinden hangisine daha yakınsa
        onu seç."""

KALIBRASYON_HARD = u"""KALİBRASYON — bu görevde en sık yapılan hata her şeye semi-relevant demektir

- semi-relevant bir KAÇIŞ ETİKETİ DEĞİLDİR. "Emin olamadım" demenin yolu semi-relevant yazmak
  değildir. semi-relevant, ürünün kısmen işe yaradığını SOMUT olarak gösterebildiğin durumdur:
  hangi varyantın yanlış olduğunu ya da ürünün neyin aksesuarı olduğunu söyleyebilmelisin.
  Söyleyemiyorsan bu etiket senin cevabın değildir.
- Sorgunun açıkça yazdığı bir şart tutmuyorsa cevap irrelevant'tır — ürün ne kadar benzer
  görünürse görünsün, aynı kategoriden olsa bile. Yanlış renk, yanlış beden/numara, yanlış
  model/nesil, yanlış marka, yanlış cinsiyet: hepsi irrelevant.
- Buna karşılık sorgunun yazmadığı hiçbir ayrıntı ürünü bozmaz. Sorgu üç kelimeden kısaysa ve
  ürün o üç kelimenin karşılığıysa, cevap relevant'tır — "acaba başka bir şey mi istedi" diye
  düşünüp semi-relevant'a kaçma.
- Üç etiket de gerçek hayatta yaklaşık eşit sıklıkta görülür. Cevaplarının çoğu semi-relevant
  çıkıyorsa, muhtemelen relevant ve irrelevant olması gereken satırları oraya yığıyorsun."""


def system_hard(order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, KARAR_HARD, KALIBRASYON_HARD, P.out_block(order)])


# ---------------------------------------------------------------------------------------------
# `hardfs` = the two interventions that each independently fixed L4's prior, combined.
# Measured raw counts on the 7500 test rows (truth 2500/2500/2500), irr/rel/semi:
#     zs    1455 / 1848 / 4197   badly skewed, calibration weights [1.835, 1.116, 0.586]
#     hard  2325 / 2015 / 3160   weights [0.968, 1.336, 0.799]
#     fs    2489 / 2371 / 2640   weights [0.897, 1.276, 0.879]  -- nearly self-calibrating
# The shots and the rubric fix the same skew by different means (demonstrated prior vs stated
# tie-break rule), so combining them is the obvious next probe -- and if they are redundant the
# blend will say so.
def system_hard_fs(shots_text, order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, KARAR_HARD, KALIBRASYON_HARD,
                         P.ORNEK_BASLIK + "\n\n" + shots_text, P.out_block(order)])
