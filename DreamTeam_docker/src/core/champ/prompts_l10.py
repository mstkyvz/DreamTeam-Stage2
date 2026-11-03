# -*- coding: utf-8 -*-
u"""L10 -- rubrics that WIDEN `semi-relevant`.

WHY THIS FILE EXISTS (F136, measured, not guessed).  Over the 7,500 holdout rows the ten 31B
prompt arms agree on 68.9% of rows at 0.9522 accuracy.  Of the 247 rows where all ten agree AND
all ten are wrong, **171 (69%) are gold `semi-relevant` that every arm called `irrelevant`.**
The model is systematically too harsh at exactly one boundary.

F137 then showed the fix cannot come from the decision rule: a per-class log-offset plus a
vote-fraction correction, aimed squarely at those 171 rows, was worth **+0.0001**.  Count-pinning
already absorbs any monotone per-class shift, so the harshness is in the *ranking*, not in the
class priors -- the information is absent from the model's output and has to be put back in by
the prompt.

Every block below is a diff against L3's `ETIKETLER`/`KARAR`/`KALIBRASYON` (used verbatim by the
`zs`/`fs` arms) and L4's `KARAR_HARD`/`KALIBRASYON_HARD` (used by `hard`/`hardfs`/`catfs_hard`,
the best single arm at 0.8825).  Keeping the role, the label header and the output block
byte-identical means the diff between an L10 arm and its L9 ancestor is the boundary rule and
nothing else.

The five rubrics form a LADDER from mild to aggressive, deliberately, because nobody knows how
far the boundary should move and one point on the line is not a measurement:

  tie   -- `hard` verbatim + ONE tie-break sentence.  Isolates the tie-break from everything else.
  soft  -- L3 family, `irrelevant` narrowed to "you must be able to NAME the violated written
           constraint", `semi` made the default.
  wide  -- the aggressive one.  A secondary attribute mismatch (colour / pattern / sleeve /
           sub-model) on a matching product type becomes SEMI, not IRRELEVANT.  This is the
           literal statement of the F136 error cell.
  type  -- procedural gate rather than a criteria list: question 1 is "is the main product type
           the same?", and if the answer is yes `irrelevant` is REMOVED from the option set.
  l7soft-- L7's `system_soft`, verbatim, written for this exact purpose and never run.

`wide` and `type` overlap in intent but not in mechanism (a rewritten criterion vs a removed
option), and F130 says mechanism-level differences are what make arms worth blending.
"""
import prompts_l3 as P
import prompts_l4_extra as PX
import prompts_l7 as L7


# =============================================================================================
# 1) `tie` -- the minimal intervention.  L4's hard rubric with ONE sentence appended.
# If this moves the score, the boundary is a tie-break and nothing more needs saying; if it does
# not, the harshness is baked deeper into the criteria and only `soft`/`wide` can reach it.
# =============================================================================================
KALIBRASYON_TIE = PX.KALIBRASYON_HARD + u"""
- SON KURAL — BERABERLİK BOZMA. semi-relevant ile irrelevant arasında kararsız kaldıysan,
  yani ürünün neden yanlış olduğunu tek bir cümleyle net söyleyemiyorsan, semi-relevant yaz.
  Bu ikisi arasındaki her tereddüt semi-relevant lehine çözülür."""


def system_tie(order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, PX.KARAR_HARD, KALIBRASYON_TIE, P.out_block(order)])


# =============================================================================================
# 2) `soft` -- L3 family, `irrelevant` narrowed by a NAMEABILITY test.
# The rubric text is unchanged except that S2 now demands the violated constraint be quotable
# from the query, and the calibration block states the base rate the model is getting wrong.
# =============================================================================================
KARAR_SOFT = u"""KARAR SIRASI — sırayla sor, ilk "evet"te dur

S1. Sorgu hangi şartları AÇIKÇA YAZIYOR? Sorguların çoğu üç kelimeden kısadır ve yalnızca bir
    ÜRÜN TİPİ söyler. Sorguda YAZMAYAN hiçbir şey şart değildir — sorgu renk demediyse rengin
    farklı olması ihlal sayılmaz, sorgu adet demediyse paket adedi ihlal sayılmaz, sorgu model
    demediyse alt-modelin farklı olması ihlal sayılmaz.

S2. Ürün, sorgunun istediğinden BAŞKA BİR İHTİYACA hizmet eden farklı bir ÜRÜN TİPİ mi?
    (mont istenmiş, kazak gelmiş; sırt çantası istenmiş, cüzdan gelmiş)
      → EVET ise: irrelevant.
      → HAYIR ise irrelevant yolu KAPANIR; S3'e geç. Ana ürün tipi tutuyorsa cevap ya relevant
        ya semi-relevant'tır.

S3. Sorgunun AÇIKÇA YAZDIĞI bir şart tamamen çiğneniyor mu? Bu şartı sorgudan KELİMESİ KELİMESİNE
    alıntılayabilmelisin ve üründe onun tam zıddı olmalı. ("kırmızı elbise" → siyah elbise;
    "erkek pantolon" → kadın pantolon; "samsung a54 kılıfı" → iphone 13 kılıfı)
      → EVET ise: irrelevant.
      → Şartı sorgudan alıntılayamıyorsan, ya da ihlal "tam zıddı" değil "biraz farklı" ise,
        cevap irrelevant DEĞİLDİR.

S4. Ürün sorgunun istediği şeyin ta kendisi ve bütün açık şartlar tutuyor mu?
      → EVET ise: relevant.

S5. Geriye kalan HER ŞEY semi-relevant'tır. Bu geniş bir banttır: doğru aileden istenmeyen
    varyant, farklı paket/set/ölçü, yakın akraba alt-tür, ikincil bir stil/nitelik farkı,
    sorgudaki ürünün aksesuarı/parçası/tamamlayıcısı, ya da sorgunun istediğinin daha genel
    veya daha özel bir alt-türü."""

KALIBRASYON_SOFT = u"""KALİBRASYON — ölçülmüş hata: bu görevde en sık yapılan hata, semi-relevant
olması gereken satırlara irrelevant demektir. Bu hata bütün hataların üçte ikisidir.

- Üç etiket de EŞİT SIKLIKTADIR: satırların tam üçte biri relevant, üçte biri semi-relevant,
  üçte biri irrelevant. Cevaplarının üçte birinden azı semi-relevant çıkıyorsa, semi olması
  gereken satırları irrelevant'a yığıyorsun demektir.
- irrelevant DAR bir etikettir ve HAK EDİLMESİ gerekir. Onu yazmadan önce şu cümleyi
  kurabilmelisin: "Sorgu ___ yazmış, ürün ise ___." Boşlukların ilkini sorgudan kelimesi
  kelimesine alıntılayamıyorsan cevap irrelevant değildir.
- Ürünün "tam olarak aranan şey olmaması" irrelevant için YETMEZ. Alıcı ürünü görünce
  "bu tam istediğim değil ama işime yarayabilir" diyorsa cevap semi-relevant'tır.
- Aksesuar, kılıf, yedek parça, filtre, aparat, uyumlu ürün ve tamamlayıcılar irrelevant
  DEĞİLDİR; semi-relevant'tır.
- Aynı ürün tipinin farklı varyantı (takım/tekil, set/tek parça, farklı ölçü, farklı paket,
  yakın akraba alt-tür, farklı kol/yaka/kesim, farklı alt-model) irrelevant DEĞİLDİR;
  semi-relevant'tır.
- KARARSIZ KALIRSAN semi-relevant seç. Bu görevin varsayılan etiketi semi-relevant'tır;
  irrelevant somut ve alıntılanabilir bir gerekçe ister."""


def system_soft(order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, KARAR_SOFT, KALIBRASYON_SOFT, P.out_block(order)])


def system_softfs(shots_text, order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, KARAR_SOFT, KALIBRASYON_SOFT,
                         P.ORNEK_BASLIK + "\n\n" + shots_text, P.out_block(order)])


# =============================================================================================
# 3) `wide` -- the aggressive arm.  The LABEL DEFINITIONS themselves are rewritten, not just the
# decision order: L3's `ETIKETLER` files colour/pattern violations under `irrelevant`, and this
# block moves them under `semi-relevant` whenever the product type matches.  This is the literal
# statement of the F136 error cell and is deliberately the most extreme point on the ladder --
# if the true boundary is anywhere between `hard` and here, some arm on this ladder brackets it.
# =============================================================================================
ETIKETLER_WIDE = u"""ÜÇ ETİKET

1) relevant — Ürün, sorgunun istediği şeyin ta kendisi. Sorguda açıkça yazan her şart üründe de
   tutuyor. Sorgunun yazmadığı ayrıntıların ne olduğu önemli değildir.

2) semi-relevant — EN GENİŞ ve EN SIK KAÇIRILAN etiket. Ürün sorgunun ANA ÜRÜN TİPİNİ
   karşılıyor ama tam istenen şey değil. Dört kalıbı vardır:

   (A) DOĞRU ÜRÜN AİLESİ, İSTENMEYEN VARYANT — ana ürün tipi tutuyor ama alıcının istediği
       varyant değil: takım istenirken tek parça, set istenirken tekil, farklı paket/ölçü/form.
         "tek kişilik yatak örtüsü takımı" → tek parça yatak örtüsü
         "kablosuz kulaklık" → kablolu kulaklık
   (B) TAMAMLAYICI / AKSESUAR / YEDEK PARÇA — sorgudaki ürünün kendisi değil, ona ait ürün.
         "iphone 15" → iPhone 15 uyumlu kılıf; "kahve makinesi" → filtre kağıdı
   (C) DOĞRU ÜRÜN TİPİ, FARKLI İKİNCİL NİTELİK — ana ürün tipi tutuyor ama ikincil bir nitelik
       (renk, desen, kol tipi, yaka, kesim, materyal, stil, alt-model, nesil) sorgudakinden
       farklı. BU DURUM irrelevant DEĞİL, semi-relevant'tır — alıcı doğru rafta doğru ürün
       tipine bakıyor, sadece istediği varyant değil.
         "puantiyeli gömlek" → çizgili gömlek; "uzun kollu tişört" → kısa kollu tişört
   (D) DAHA GENEL ya da DAHA ÖZEL ALT-TÜR — ürün sorgunun istediğinin üst ya da alt kümesi.

3) irrelevant — DAR bir etiket. Yalnızca iki durumda:
     - ÜRÜN TİPİ FARKLI: ürün bambaşka bir ihtiyaca hizmet ediyor. Alıcının aradığı raf bu değil.
       "mont" → kazak; "sırt çantası" → cüzdan; "kışlık bot" → babet
     - SORGUNUN KİLİT NİTELİĞİ TAMAMEN İHLAL EDİLMİŞ: sorgunun bütün anlamını taşıyan tek şart
       tam zıddına dönmüş. "erkek kot pantolon" → kadın kot pantolon; "samsung a54 kılıfı" →
       iphone 13 kılıfı. Buradaki şart, kaldırıldığında sorgudan geriye anlamlı bir şey kalmayan
       şarttır (cinsiyet, marka/model uyumluluğu).
   Bunların dışında kalan hiçbir uyuşmazlık irrelevant değildir."""

KARAR_WIDE = u"""KARAR SIRASI — sırayla sor, ilk "evet"te dur

S1. ANA ÜRÜN TİPİ AYNI MI? Yani ürün, sorgunun aradığı ihtiyaca hizmet eden şeyin kendisi mi
    (ya da onun bir varyantı / aksesuarı / parçası)?
      → HAYIR ise: irrelevant. Karar burada biter.
      → EVET ise: bu satır artık irrelevant OLAMAZ (tek istisna S2). S2'ye geç.

S2. Sorgunun KİLİT niteliği — kaldırılsa sorgunun anlamı yok olacak olan tek şart, yani
    cinsiyet, yaş grubu ya da marka/model uyumluluğu — tam zıddına ihlal edilmiş mi?
      → EVET ise: irrelevant.
      → HAYIR ise: S3.

S3. Ürün sorgunun istediği şeyin ta kendisi mi ve sorgunun AÇIKÇA YAZDIĞI her şart tutuyor mu?
      → EVET ise: relevant.

S4. Buraya geldiysen cevap semi-relevant'tır. Ana ürün tipi tutuyor ama bir şey farklı: renk,
    desen, kol/yaka/kesim, materyal, stil, alt-model, paket/set, ölçü, ya da ürün sorgudakinin
    aksesuarı. Hepsi semi-relevant'tır."""

KALIBRASYON_WIDE = u"""KALİBRASYON

- Üç etiket de eşit sıklıktadır (her biri satırların üçte biri). Ölçülmüş hata: modeller
  semi-relevant olması gereken satırlara irrelevant diyor. Bu, bütün hataların üçte ikisi.
- Tek soruya indir: ALICI DOĞRU RAFA MI BAKIYOR? Raf doğruysa (ürün tipi tutuyorsa) cevap
  relevant ya da semi-relevant'tır — renk, desen, kesim, alt-model ne kadar farklı olursa olsun.
  Yanlış raf ise irrelevant'tır.
- "Yeşil topuklu ayakkabı" sorgusuna gelen yeşil SPOR ayakkabı irrelevant'tır: spor ayakkabı
  başka bir raftır. Ama "puantiyeli gömlek" sorgusuna gelen düz gömlek semi-relevant'tır:
  gömlek rafı doğru, sadece desen farklı.
- KARARSIZ KALIRSAN semi-relevant seç."""


def system_wide(order='label'):
    return u"\n\n".join([P.ROL, ETIKETLER_WIDE, KARAR_WIDE, KALIBRASYON_WIDE, P.out_block(order)])


# =============================================================================================
# 4) `type` -- the same widening delivered as a REMOVED OPTION rather than a rewritten criterion.
# Question 1 is the product-type gate and, when it passes, the answer set is stated as a
# two-element set with `irrelevant` explicitly absent.  A model cannot pick an option it has been
# told is not on the table, which is a different failure surface from a criterion it can
# reinterpret -- so this arm should decorrelate with `wide` even where they agree in intent.
# L3's ETIKETLER is kept BYTE-IDENTICAL here so the only diff against `zs` is the gate.
# =============================================================================================
KARAR_TYPE = u"""KARAR YÖNTEMİ — İKİ AŞAMALI KAPI

AŞAMA 1 — ÜRÜN TİPİ KAPISI. Önce ve yalnızca şu soruyu sor:
    "Bu ürün, sorgunun aradığı ürün tipinin kendisi mi, bir varyantı mı, yoksa aksesuarı/
     parçası/tamamlayıcısı mı? Yoksa bambaşka bir ürün tipi mi?"

  Ürünün rengi, deseni, ölçüsü, kesimi, alt-modeli bu soruda HİÇ dikkate alınmaz. Yalnızca
  "bu ne türden bir nesne" sorusu sorulur.

  1a) Bambaşka bir ürün tipiyse (alıcı bu rafta aradığını bulamaz) → irrelevant. BİTTİ.
  1b) Aynı ürün tipiyse / varyantıysa / aksesuarıysa → GEÇTİ. Bu satır için artık
      SEÇENEKLERİN YALNIZCA İKİ TANEDİR: relevant veya semi-relevant.
      irrelevant BU SATIR İÇİN ARTIK BİR SEÇENEK DEĞİLDİR — tek istisna aşağıdaki S2.

AŞAMA 2 — yalnızca 1b'den geldiysen.
  S2. Sorguda AÇIKÇA yazılı bir CİNSİYET, YAŞ GRUBU ya da MARKA/MODEL var mı ve ürün bunun tam
      zıddı mı? ("erkek" → kadın ürünü; "samsung a54 kılıfı" → iphone kılıfı)
        → EVET ise: irrelevant. (Kapıyı yalnızca bu üç alan açar.)
        → HAYIR ise: S3.
  S3. Ürün sorgunun istediği şeyin ta kendisi ve sorgunun yazdığı bütün şartlar tutuyor mu?
        → EVET ise: relevant.
        → HAYIR ise: semi-relevant.

Aşama 2'de renk, desen, kesim, materyal, kol tipi, ölçü, paket, alt-tür farkları irrelevant'a
YOL AÇMAZ; bunlar S3'ü düşürür ve cevabı semi-relevant yapar."""

KALIBRASYON_TYPE = u"""KALİBRASYON

- Üç etiket de eşit sıklıktadır. Ölçülmüş hata: semi-relevant olması gereken satırlara
  irrelevant deniyor; bu bütün hataların üçte ikisi. Kapı kuralı bu hatayı kapatmak içindir.
- Aşama 1'i geçen bir satıra irrelevant yazmak KURAL İHLALİDİR (S2'nin üç alanı dışında).
- Aksesuar / kılıf / yedek parça / filtre / aparat / uyumlu ürün Aşama 1'i GEÇER.
- Kararsız kaldığın her yerde semi-relevant seç."""


def system_type(order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, KARAR_TYPE, KALIBRASYON_TYPE, P.out_block(order)])


# =============================================================================================
# 5) `l7soft` -- L7's `system_soft`, verbatim, imported rather than copied so it cannot drift.
# It is a BARE arm (no `gerekce` field in the output block) and carries L7's own label block, so
# it differs from the rest of the ladder on two axes at once.  That is not a design flaw here:
# the brief asks for it to be run as written, and its independence is what makes it a useful
# blend member.
# =============================================================================================
def system_l7soft(order='label'):
    return L7.system_soft()


# CONTROLS.  `hard` is L9's rubric verbatim; running it here is what makes every L10 timing
# comparable to the F138 budget frontier, which was measured on an H100 NVL and not on this
# L40S. One control pass gives the exact hardware ratio, so the ladder's wall-clock numbers can
# be read against the 180 s budget instead of being uninterpretable.

# =============================================================================================
# 6) `ultra` -- deliberately PAST the useful point, on the coordinator's measurement.
#
# `wide` scored 0.8777 alone against the 0.8801 reference (slightly over-shot: irr 2353 against a
# true 2500) but 0.8996 as a HELPER over the 31B and lifted the F137 blend to 0.9097. The two
# facts are one fact: `wide` is wrong in the OPPOSITE direction from every other arm, which is
# precisely the missing error direction F136 identified. Over-shooting did not cost anything in
# the helper role -- it is what created the decorrelation.
#
# So this arm asks the obvious next question: if one step past the boundary helped, does two?
# `ultra` removes the last escape `wide` kept. In `wide`, a reversed gender / age-group /
# brand-model still buys `irrelevant`; here NOTHING does except a genuinely different product
# type. It is expected to be bad alone. That is not the point -- a cheap arm that is wrong in a
# direction nothing else is wrong in is worth more to the blend than a mediocre arm that is wrong
# where everything else already is. If it is worth nothing, that is a one-pass answer.
ETIKETLER_ULTRA = u"""ÜÇ ETİKET

1) relevant — Ürün, sorgunun istediği şeyin ta kendisi ve sorgunun AÇIKÇA yazdığı her şart üründe
   de tutuyor. Alıcı ürünü görünce "evet, aradığım tam olarak bu" der.

2) semi-relevant — BU GÖREVİN VARSAYILAN ETİKETİDİR ve en geniş banttır. Ürün, alıcının aradığı
   İHTİYACIN çevresinde bir yerde duruyor ama tam istediği şey değil. Tek ölçüt şudur:
   alıcı bu ürünü arama sonuçlarında görünce "bu da bir seçenek olabilir" ya da "bu benim
   aradığım rafta ama istediğim değil" diyorsa → semi-relevant.
   Bu bandın içine giren her şey:
     - doğru ürün tipi, farklı renk / desen / kesim / kol / yaka / materyal / stil
     - doğru ürün tipi, farklı alt-model / nesil / seri / beden sınıfı
     - doğru ürün tipi, farklı MARKA
     - doğru ürün tipi, farklı CİNSİYET ya da YAŞ GRUBU
     - takım/tekil, set/tek parça, farklı paket, farklı ölçü, farklı adet
     - sorgudakinin daha genel ya da daha özel bir alt-türü
     - sorgudaki ürünün aksesuarı, kılıfı, yedek parçası, filtresi, aparatı, tamamlayıcısı
   Yukarıdakilerin HEPSİ semi-relevant'tır. Hiçbiri irrelevant değildir.

3) irrelevant — EN DAR etiket. YALNIZCA tek bir durumda: ürün BAŞKA BİR İHTİYACA hizmet ediyor,
   yani alıcının aradığı raf bu değil. Ürünün ne olduğunu söylediğinde sorgunun aradığı nesne
   sınıfının dışına düşüyorsa irrelevant'tır.
     "mont" → kazak (biri dış giyim biri üst giyim: farklı ihtiyaç)
     "sırt çantası" → cüzdan
     "kışlık bot" → babet
     "topuklu ayakkabı" → spor ayakkabı (topuk ayrı bir nesne sınıfıdır)
   Ürünün ne olduğu sorgunun aradığı nesne sınıfının İÇİNDEYSE, geri kalan hiçbir uyuşmazlık
   irrelevant yapmaz — o satır semi-relevant'tır."""

KARAR_ULTRA = u"""KARAR — TEK SORU

Yalnızca şunu sor: BU ÜRÜN NE TÜR BİR NESNE, ve bu nesne sınıfı sorgunun aradığı nesne sınıfının
kendisi mi (ya da onun bir varyantı / aksesuarı / parçası) mı?

  - HAYIR, bambaşka bir nesne sınıfı → irrelevant. Karar biter.
  - EVET → irrelevant SEÇENEĞİ BU SATIR İÇİN TAMAMEN KAPANDI. Geriye iki seçenek kaldı:
      * sorgunun yazdığı HER şart tutuyor ve ürün istenen şeyin ta kendisi → relevant
      * başka her durum → semi-relevant

Renk, desen, kesim, materyal, marka, model, nesil, cinsiyet, yaş grubu, beden, paket, ölçü:
bunların hiçbiri tek başına irrelevant'a yol açmaz. Hepsi relevant ile semi-relevant arasında
karar verdirir."""

KALIBRASYON_ULTRA = u"""KALİBRASYON

- irrelevant bu görevin EN DAR etiketidir ve yalnızca "yanlış raf" demektir.
- Kararsız kaldığın HER satırda semi-relevant yaz. İstisnasız.
- Ürünün sorguya kelime olarak ne kadar benzediği ya da benzemediği önemli değildir; önemli olan
  yalnızca nesne sınıfının aynı olup olmadığıdır."""


def system_ultra(order='label'):
    return u"\n\n".join([P.ROL, ETIKETLER_ULTRA, KARAR_ULTRA, KALIBRASYON_ULTRA,
                          P.out_block(order)])



# =============================================================================================
# 7) `shwide` -- the SAME widening, delivered through L3's buyer frame instead of the expert one.
#
# The coordinator's measurement is that `wide` pays because it is wrong in a direction nothing
# else is wrong in. That is a decorrelation argument, and decorrelation is bought by MECHANISM,
# not by intent: F130 measured that a second architecture over one encoder is worth +0.003..+0.0055
# precisely because it fails elsewhere. L9's `sh` (buyer frame) is the structurally most different
# prompt family in the pool -- weakest alone at 0.8564 but with the most balanced raw counts of
# any L9 arm (2474/2400/2626) -- so putting the widened boundary inside THAT frame should land in
# a third place again, rather than next to `wide`.
#
# Every block is L3's buyer wording with only the boundary moved, so the diff against `sh` is the
# same diff `wide` is against `zs`.
UC_TEPKI_WIDE = u"""ÜÇ OLASI TEPKİN VAR

- "Aradığım şey tam olarak buydu." -> relevant
  Ürün, yazdığın kelimelerin karşılığı olan şeyin kendisi ve yazdığın hiçbir şarta ters düşmüyor.

- "Bu tam istediğim değil ama aynı işe bakıyor / işime yarayabilir." -> semi-relevant
  BU EN SIK TEPKİNDİR ve bandı geniştir. Aradığın RAFTA duran ama istediğin olmayan her şey
  buraya girer:
    * doğru ürün ama farklı renk, desen, kesim, kol, yaka, materyal, stil;
    * doğru ürün ama farklı alt-model, nesil, seri;
    * doğru şeyin farklı bir versiyonu (takım isterken tekil, set isterken tek parça, yakın
      akraba bir alt-tür, farklı paket/ölçü/form);
    * ürün aradığın şeyin kendisi değil ama ona ait: kılıfı, yedek parçası, aparatı, filtresi.

- "Hayır, bu yanlış ürün, ben bunu istemedim." -> irrelevant
  Bunu YALNIZCA ürün bambaşka bir ihtiyaca hizmet ediyorsa dersin — yani aradığın raf bu değil
  ("mont" yazdın, kazak geldi; "sırt çantası" yazdın, cüzdan geldi) — ya da yazdığın cinsiyet /
  yaş grubu / marka-model tam tersine dönmüşse. Bunların dışında "yanlış ürün" DEMEZSİN."""

ALICI_KALIBRASYON_WIDE = u"""KENDİNİ TUT — "bu yanlış ürün" demek için sebep göstermek zorundasın

Ürünün aradığın şeyin tam karşılığı olmaması, "yanlış ürün" demen için yetmez. Tek soru şu:
DOĞRU RAFA MI BAKIYORUM? Raf doğruysa tepkin en fazla "tam istediğim değil ama aynı işe bakıyor"
olabilir.
  - aradığın ürünün kılıfı/parçası/aparatı yanlış ürün değildir;
  - aynı şeyin farklı varyantı (takım/tekil, set/tek parça, farklı ölçü, farklı paket, yakın
    akraba bir alt-tür) yanlış ürün değildir;
  - doğru ürünün farklı rengi, deseni, kesimi, kolu, materyali ya da alt-modeli yanlış ürün
    DEĞİLDİR — istediğin varyant değildir, o kadar;
  - sorguna yazmadığın bir ayrıntının farklı çıkması yanlış ürün değildir;
  - sorgun sadece bir marka adıysa o markanın her ürünü işine yarayabilir.
EMİN OLAMIYORSAN "tam istediğim değil ama aynı işe bakıyor" (semi-relevant) de. Bu üç tepkiyi
yaklaşık eşit sıklıkta veriyor olmalısın."""


def system_shwide(order='label'):
    out = (P.CIKTI_ALICI_REASON_FIRST if order == 'reason' else P.CIKTI_ALICI_LABEL_FIRST)
    return u"\n\n".join([P.ROL_ALICI, UC_TEPKI_WIDE, ALICI_KALIBRASYON_WIDE, out])



# =============================================================================================
# 8) `wide2` -- the coordinator's TWO-GATE rule, from statistics they derived on their own
# calibration set (no labelled rows were handed over; the numbers below are theirs):
#
#   query's product-type word appears in the title (n=195):  irrelevant only  11.3%
#   it does not (n=305):                                     irrelevant only  47.5%
#   mean query/title token overlap: relevant 0.565, semi 0.411, irrelevant 0.134
#
# Two corrections to `wide` follow, and they pull in OPPOSITE directions, which is why this is a
# two-gate rule rather than a shifted threshold:
#
#  (1) When the product type matches, `irrelevant` is almost never right (11.3%), so gate 1
#      eliminates it entirely -- `wide` already did this, but it kept an escape for reversed
#      gender / brand-model and that escape is not supported by the statistic. The subtlety the
#      coordinator flags is that the type test must be done on MEANING, not surface form: Turkish
#      inflection makes "bikini alti" and "bikini alt" different strings and the same type. Their
#      own naive last-word proxy missed those, and the model plausibly does too, so the rubric
#      says it out loud.
#  (2) When the product type does NOT match, `irrelevant` is still only 47.5% -- the other half is
#      semi (31.8%) and even relevant (20.7%). So a type mismatch is NOT sufficient for
#      `irrelevant`, which is exactly what `wide` (and every earlier rubric) assumed. Gate 2 asks
#      the need question instead.
#
# Net effect against `wide`: gate 1 is slightly more permissive (the gender/brand escape is gone)
# and gate 2 is much more permissive. Both push toward `semi`, so `wide2` risks the `relevant`
# bleed that killed `ultra` (rel 2201, 374 rows lost out of `relevant`). The RELEVANT clause is
# therefore deliberately left tight and given its own explicit guard -- that guard is not in the
# coordinator's rule, it is a correction for the failure `ultra` and `shwide` measured here.
ETIKETLER_WIDE2 = u"""ÜÇ ETİKET

1) relevant — Ürün, sorgunun istediği şeyin ta kendisi ve sorgunun AÇIKÇA yazdığı şartların hepsi
   tutuyor. Sorgunun yazmadığı ayrıntılar önemli değildir.

2) semi-relevant — EN GENİŞ etiket. İki ayrı yoldan buraya gelinir:
   (A) Ürün tipi sorgunun aradığı tiple AYNI ama ürün tam istenen şey değil: farklı renk, desen,
       kesim, kol, yaka, materyal, stil, alt-model, nesil, marka, cinsiyet/yaş, beden, paket,
       set/tekil, ölçü, adet — ya da ürün o tipin aksesuarı/parçası/kılıfı/tamamlayıcısı.
   (B) Ürün tipi FARKLI GÖRÜNÜYOR ama ürün, sorgunun karşıladığı İHTİYACI karşılıyor: alıcı
       aradığı işi bu ürünle de görebilir, ya da aynı ihtiyaç için değerlendirebileceği bir
       alternatif. Bu yol yenidir ve sık kullanılmalıdır.

3) irrelevant — EN DAR etiket. Yalnızca ürün sorgunun ihtiyacıyla İLGİSİZ BİR İHTİYACA hizmet
   ediyorsa. Yani alıcı ürüne bakıp "bunun benim aradığım şeyle hiçbir ilgisi yok" diyorsa.
   Ürün tipinin farklı olması TEK BAŞINA yetmez — farklı tip + ilgisiz ihtiyaç gerekir."""

KARAR_WIDE2 = u"""KARAR — İKİ KAPI. Sırayla geç, atlama.

KAPI 1 — ÜRÜN TİPİ (ANLAMCA) AYNI MI?
  Sorgunun aradığı nesne türü ile ürünün nesne türünü KÖK ANLAMIYLA karşılaştır.
  ⚠️ Yüzey biçimine BAKMA. Türkçe çekim ekleri, iyelik ekleri, çoğul ekleri ve eş anlamlı
     kelimeler tip farkı SAYILMAZ:
       "bikini altı" ile "bikini alt" AYNI tiptir;
       "çocuk ayakkabısı" ile "çocuk ayakkabı" AYNI tiptir;
       "telefon kılıfı" ile "cep telefonu kılıfı" AYNI tiptir;
       "sweatshirt" ile "sweat" AYNI tiptir.
     Kelimeler farklı yazılıyor diye tip farklı deme; nesnenin kendisi aynı mı ona bak.

  KAPI 1'DEN GEÇTİYSE (tip aynı):
    → irrelevant BU SATIR İÇİN ELENDİ. Başka hiçbir uyuşmazlık onu geri getirmez: renk, desen,
      marka, model, cinsiyet, yaş grubu, beden — hiçbiri.
    → Karar yalnızca relevant ile semi-relevant arasındadır:
        * sorgunun AÇIKÇA yazdığı şartların HEPSİ tutuyor ve ürün istenen şeyin ta kendisi
          → relevant
        * aksi halde → semi-relevant
    → Buradan sonra KAPI 2'ye BAKMA, karar bitti.

KAPI 2 — tip farklı görünüyorsa: AYNI İHTİYACI MI KARŞILIYOR?
  "Bu ürün, alıcının bu sorguyu yazarken karşılamak istediği ihtiyacı karşılar mı, ya da o ihtiyaç
   için değerlendirilebilecek bir alternatif mi?"
    → EVET ise: semi-relevant.
    → HAYIR, ürün bambaşka ve ilgisiz bir ihtiyaca hizmet ediyorsa: irrelevant."""

KALIBRASYON_WIDE2 = u"""KALİBRASYON

- Üç etiket de yaklaşık eşit sıklıktadır (her biri satırların üçte biri).
- Ölçülmüş: ürün tipi tuttuğunda cevabın irrelevant olma ihtimali yalnızca ~%11'dir. Kapı 1'i
  geçen bir satıra irrelevant yazıyorsan neredeyse kesinlikle yanılıyorsun.
- Ölçülmüş: ürün tipi tutmadığında bile cevabın irrelevant olma ihtimali yalnızca ~%48'dir.
  Yani "tip farklı" tek başına irrelevant demek DEĞİLDİR; ikinci kapıyı sormadan karar verme.
- ⚠️ AMA relevant'ı BOŞALTMA. Sorgunun yazdığı her şart tutuyorsa ve ürün istenen şeyin ta
  kendisiyse cevap relevant'tır — "acaba başka bir varyant mı istedi" diye düşünüp semi-relevant'a
  kaçma. semi-relevant'ın genişlemesi irrelevant'tan gelmelidir, relevant'tan DEĞİL.
- Kararsız kaldığında semi-relevant seç."""


def system_wide2(order='label'):
    return u"\n\n".join([P.ROL, ETIKETLER_WIDE2, KARAR_WIDE2, KALIBRASYON_WIDE2,
                          P.out_block(order)])



# =============================================================================================
# 9) `wide3` -- `wide2` with gate 2 re-calibrated, because gate 2 as written over-fires.
#
# MEASURED: `wide2` returned irr/rel/semi = 1696/2434/3370 against a truth of 2500/2500/2500.
# That is the largest over-shoot of any arm here -- further out than `ultra` (2030) and far past
# `wide` (2353) -- even though the coordinator's statistics were meant to CORRECT `wide`'s
# over-promotion. The two gates are not equally to blame and the split is diagnosable:
#
#   gate 1 (semantic type match removes `irrelevant`) is supported by their number: when the
#           product type matches, `irrelevant` is right only 11.3% of the time. Keep it.
#   gate 2 ("different type but serves the same need -> semi") is where the collapse is. Their
#           statistic says a type mismatch is `irrelevant` 47.5% of the time -- roughly a coin
#           flip -- but "does it serve the same need?" is a question this model answers YES to
#           almost always, because at a loose enough reading every product in a department serves
#           a related need. A qualitative question cannot hit a 47.5% base rate.
#
# So gate 2 is restated as a NARROW test with the base rate stated numerically and the default
# flipped to `irrelevant`: on a type mismatch the answer is `irrelevant` unless the product is
# specifically an accessory/part/complement of the queried product, or a direct substitute a
# buyer would weigh instead of it. Gate 1 keeps `wide2`'s inflection wording verbatim, which is
# the part of the coordinator's rule that is unambiguously right and that no earlier arm had.
KARAR_WIDE3 = u"""KARAR — İKİ KAPI. Sırayla geç, atlama.

KAPI 1 — ÜRÜN TİPİ (ANLAMCA) AYNI MI?
  Sorgunun aradığı nesne türü ile ürünün nesne türünü KÖK ANLAMIYLA karşılaştır.
  ⚠️ Yüzey biçimine BAKMA. Türkçe çekim ekleri, iyelik ekleri, çoğul ekleri ve eş anlamlı
     kelimeler tip farkı SAYILMAZ:
       "bikini altı" ile "bikini alt" AYNI tiptir;
       "çocuk ayakkabısı" ile "çocuk ayakkabı" AYNI tiptir;
       "telefon kılıfı" ile "cep telefonu kılıfı" AYNI tiptir;
       "sweatshirt" ile "sweat" AYNI tiptir.
     Kelimeler farklı yazılıyor diye tip farklı deme; nesnenin kendisi aynı mı ona bak.

  KAPI 1'DEN GEÇTİYSE (tip aynı):
    → irrelevant BU SATIR İÇİN ELENDİ. Renk, desen, marka, model, cinsiyet, yaş, beden — hiçbiri
      onu geri getirmez.
    → Karar yalnızca relevant ile semi-relevant arasındadır:
        * sorgunun AÇIKÇA yazdığı şartların HEPSİ tutuyor ve ürün istenen şeyin ta kendisi
          → relevant
        * aksi halde → semi-relevant
    → Karar bitti, KAPI 2'ye BAKMA.

KAPI 2 — tip FARKLI ise. Burada VARSAYILAN CEVAP irrelevant'tır.
  Ölçülmüş: ürün tipi tutmadığında satırların yaklaşık YARISI irrelevant'tır. Yani tip farklıysa
  bol keseden semi-relevant dağıtma; semi-relevant'ı HAK ETTİR.
  Yalnızca şu iki dar durumda semi-relevant yaz:
    (a) Ürün, sorgudaki ürünün AKSESUARI / PARÇASI / KILIFI / FİLTRESİ / APARATI /
        TAMAMLAYICISI. ("iphone 15" → iPhone 15 kılıfı; "kahve makinesi" → filtre kağıdı)
    (b) Ürün, sorgudakinin DOĞRUDAN YERİNE GEÇEBİLECEK bir alternatifi: alıcı ikisini yan yana
        koyup "hangisini alsam" diye düşünür. ("ahşap jaluzi" → plise perde; "kablosuz kulaklık"
        → kablolu kulaklık)
  Bu ikisinden birini SOMUT olarak gösteremiyorsan cevap irrelevant'tır. "Aynı reyonda", "benzer
  bir ihtiyaca hizmet ediyor olabilir" ya da "alıcının işine yarayabilir" YETMEZ."""

KALIBRASYON_WIDE3 = u"""KALİBRASYON

- Üç etiket de yaklaşık eşit sıklıktadır (her biri satırların üçte biri).
- Kapı 1'i geçen bir satıra irrelevant yazıyorsan neredeyse kesinlikle yanılıyorsun (ölçülmüş:
  tip tuttuğunda irrelevant olma ihtimali ~%11).
- Kapı 2'de ise tersi geçerlidir: tip tutmuyorsa irrelevant ihtimali ~%48'dir, yani yazı-tura.
  Orada semi-relevant somut bir aksesuar/parça ilişkisi ya da doğrudan bir ikame ilişkisi ister.
- ⚠️ relevant'ı BOŞALTMA. Sorgunun yazdığı her şart tutuyorsa ve ürün istenen şeyin ta
  kendisiyse cevap relevant'tır; semi-relevant'a kaçma. semi-relevant'ın genişlemesi
  irrelevant'tan gelmelidir, relevant'tan DEĞİL."""


def system_wide3(order='label'):
    return u"\n\n".join([P.ROL, ETIKETLER_WIDE2, KARAR_WIDE3, KALIBRASYON_WIDE3,
                          P.out_block(order)])



# =============================================================================================
# 10) `softtype` -- the synthesis the measurements point at, and the last untried combination.
#
# Two separate things have now been measured on this box, and they are on different axes:
#
#   CALIBRATION.  `soft` / `soft_fs` produce the raw counts closest to the true 2500/2500/2500
#                 (soft_fs: 2546/2413/2541, off by 46/87/41). Every `wide*` arm over-shoots,
#                 `wide2` worst at irr 1696.
#   MECHANISM.    The coordinator's gate-1 wording -- compare product type by ROOT MEANING, with
#                 the Turkish inflection cases spelled out -- raised the promotion ratio sharply:
#                 `wide2` moves 1325 rows at ratio 7.3 where `ultra` moved 999 at 2.7.
#
# No arm has both. `soft_fs`'s ratio is only 3.1, the second worst in the run -- it is
# well-calibrated but imprecise about WHICH rows it moves. So this arm is `soft` with exactly one
# paragraph added: the inflection warning, inserted into the step where `soft` already asks the
# product-type question. Nothing else changes -- same S1, same S3-S5, same calibration block --
# so if the counts hold and the ratio improves, the paragraph is the cause and it is portable to
# any rubric.
KARAR_SOFTTYPE = KARAR_SOFT.replace(
    u"""S2. Ürün, sorgunun istediğinden BAŞKA BİR İHTİYACA hizmet eden farklı bir ÜRÜN TİPİ mi?
    (mont istenmiş, kazak gelmiş; sırt çantası istenmiş, cüzdan gelmiş)""",
    u"""S2. Ürün, sorgunun istediğinden BAŞKA BİR İHTİYACA hizmet eden farklı bir ÜRÜN TİPİ mi?
    (mont istenmiş, kazak gelmiş; sırt çantası istenmiş, cüzdan gelmiş)
    ⚠️ Tipi KÖK ANLAMIYLA karşılaştır, yüzey biçimiyle DEĞİL. Türkçe çekim ekleri, iyelik
       ekleri, çoğul ekleri ve eş anlamlı kelimeler tip farkı SAYILMAZ:
         "bikini altı" ile "bikini alt" AYNI tiptir;
         "çocuk ayakkabısı" ile "çocuk ayakkabı" AYNI tiptir;
         "telefon kılıfı" ile "cep telefonu kılıfı" AYNI tiptir;
         "sweatshirt" ile "sweat" AYNI tiptir.
       Kelimeler farklı yazılıyor diye tip farklı deme; nesnenin kendisi aynı mı ona bak.
       Ölçülmüş: ürün tipi tuttuğunda cevabın irrelevant olma ihtimali yalnızca ~%11'dir.""")
assert KARAR_SOFTTYPE != KARAR_SOFT, 'the S2 anchor text moved; the diff would be silent'


def system_softtype(order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, KARAR_SOFTTYPE, KALIBRASYON_SOFT,
                          P.out_block(order)])



# =============================================================================================
# 11) THE DIVERSITY BATCH.  A different objective from everything above.
#
# The coordinator measured that an arm's value in the blend is INVERSELY related to its quality
# alone: standalone order wide2_fs > soft_fs > wide > wide3 > ultra > wide2, blend order
# wide3 > wide2 > wide2_fs > ultra > wide -- nearly reversed. `wide2` is 0.8218 alone (raw irr
# 1696) and the second most valuable member. So these arms are not trying to be good. They are
# trying to be wrong in directions the pool has never been wrong in, and being bad alone is
# evidence they are doing that.
#
# The pool is currently extreme in ONE direction only -- every L10 arm past `hard` widens `semi`.
# Four untouched axes:
#   A  the opposite pole: widen `irrelevant`, shrink `semi`   -> `harsh`, `harsher`
#   B  the OTHER boundary, relevant/semi, never once pushed    -> `relwide`, `relnarrow`
#   C  single-attribute tunnel vision, each wrong elsewhere    -> `gender`, `brand`
#   D  information quantity rather than criteria               -> `tiny`, `mega`

# --- axis A: the opposite pole -------------------------------------------------------------
KARAR_HARSH = u"""KARAR SIRASI

S1. Ürün, sorgunun tarif ettiği şeyin BİREBİR karşılığı mı? Sorgudaki her kelimenin karşılığı
    üründe var mı?
      → EVET ise: relevant.
S2. Değilse: uyuşmayan şeyi göster. Renk mi, desen mi, tip mi, marka mı, model mi, cinsiyet mi,
    adet mi, ölçü mü, kesim mi?
      → Uyuşmayan HERHANGİ bir şey gösterebiliyorsan: irrelevant.
S3. semi-relevant YALNIZCA tek bir durumda yazılır: ürün, sorgudaki ürünün açıkça aksesuarı,
    yedek parçası, kılıfı ya da filtresidir. Başka hiçbir durumda semi-relevant yazma."""

KALIBRASYON_HARSH = u"""KALİBRASYON — bu görevde en sık yapılan hata fazla hoşgörülü olmaktır

- Alıcı aradığını bulamadıysa ürün YANLIŞTIR. "İşine yarayabilir" diye düşünme; alıcı ne aradığını
  biliyor ve başka bir şey istemiyor.
- Aynı reyondan olmak, benzer görünmek, aynı kelimeleri taşımak hiçbir şey ifade etmez.
- semi-relevant bir KAÇIŞ değildir; yalnızca aksesuar/parça ilişkisi için ayrılmıştır.
- Şüphedeysen irrelevant seç. Bu görevin varsayılan etiketi irrelevant'tır."""


def system_harsh(order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, KARAR_HARSH, KALIBRASYON_HARSH, P.out_block(order)])


ETIKETLER_HARSHER = u"""ÜÇ ETİKET

1) relevant — Ürün, sorgunun istediği şeyin BİREBİR kendisi. Sorgudaki her kelime üründe
   karşılığını buluyor ve ürün başka hiçbir yönden sapmıyor.

2) semi-relevant — ÇOK DAR. Yalnızca: ürün, sorgudaki ürünün aksesuarı / yedek parçası /
   kılıfı / filtresi / aparatıdır. Bunun dışında hiçbir şey semi-relevant değildir.

3) irrelevant — VARSAYILAN VE EN SIK ETİKET. Ürün sorgunun istediği şeyin birebir kendisi
   değilse ve aksesuarı da değilse, irrelevant'tır. Farklı renk, farklı desen, farklı kesim,
   farklı beden, farklı paket, farklı alt-tür, farklı marka, farklı model, farklı nesil, set
   yerine tekil, tekil yerine set: HEPSİ irrelevant'tır. Alıcı istediğini bulamamıştır."""


def system_harsher(order='label'):
    return u"\n\n".join([P.ROL, ETIKETLER_HARSHER, KARAR_HARSH, KALIBRASYON_HARSH,
                          P.out_block(order)])


# --- axis B: the relevant/semi boundary, which nothing in this run has touched --------------
KALIBRASYON_RELWIDE = u"""KALİBRASYON — dikkat edilecek sınır relevant ile semi-relevant arasındadır

Sorgular kısadır ve alıcı yalnızca yazdığı şeyi şart koşar. Bu yüzden en sık yapılan hata,
aslında relevant olan satırlara semi-relevant demektir.

- Sorgunun YAZMADIĞI hiçbir ayrıntı ürünü semi-relevant yapmaz. Sorgu renk yazmadıysa ürünün
  rengi ne olursa olsun relevant'tır. Sorgu adet/paket yazmadıysa ürünün kaç adet olduğu
  önemsizdir. Sorgu alt-model yazmadıysa hangi alt-model olduğu önemsizdir.
- "Acaba başka bir varyantını mı istedi" diye düşünme. Sorgu ne yazdıysa şart odur; gerisi
  alıcının umurunda değildir.
- semi-relevant YALNIZCA sorgunun yazdığı bir şart kısmen tutmuyorsa ya da ürün sorgudakinin
  aksesuarı/parçasıysa yazılır.
- irrelevant ise yalnızca sorgunun yazdığı bir şart açıkça çiğnendiğinde yazılır.
- Şüphedeysen relevant seç."""


def system_relwide(order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, P.KARAR, KALIBRASYON_RELWIDE, P.out_block(order)])


KALIBRASYON_RELNARROW = u"""KALİBRASYON — dikkat edilecek sınır relevant ile semi-relevant arasındadır

En sık yapılan hata, aslında semi-relevant olan satırlara relevant demektir. relevant çok DAR
bir etikettir: ürün, alıcının aradığı şeyin TAM KARŞILIĞI olmalıdır.

- Ürün ile sorgu arasında en ufak bir sapma varsa — farklı bir alt-tür, farklı bir paket, farklı
  bir ölçü, farklı bir kesim, fazladan ya da eksik bir parça, farklı bir kullanım biçimi —
  cevap relevant DEĞİL, semi-relevant'tır.
- Ürünün başlığında sorgunun kelimeleri geçiyor olması relevant yapmaz; ürünün gerçekten
  aradığı şey olması gerekir.
- relevant demeden önce şunu sor: "alıcı bunu sipariş etse, kutuyu açtığında aradığı şeyin ta
  kendisini mi bulur, yoksa 'yakın ama tam değil' mi der?" İkincisi ise semi-relevant.
- irrelevant sınırı değişmedi: yalnızca sorgunun yazdığı bir şart açıkça çiğnendiğinde.
- Şüphedeysen relevant yerine semi-relevant seç."""


def system_relnarrow(order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, P.KARAR, KALIBRASYON_RELNARROW, P.out_block(order)])


# --- axis C: single-attribute tunnel vision -------------------------------------------------
# Each of these is told to decide on ONE attribute and to treat everything else as secondary.
# They are guaranteed to be wrong on the rows that attribute does not govern -- which is the
# point: two arms with disjoint blind spots decorrelate by construction.
def _attr_system(attr_block, order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, attr_block, P.out_block(order)])


ATTR_GENDER = u"""KARAR YÖNTEMİ — CİNSİYET VE YAŞ GRUBU ÜZERİNDEN

Bu satırda kararını ÖNCELİKLE cinsiyet ve yaş grubu uyumuna göre ver. Diğer her şey ikincildir.

S1. Sorgu bir cinsiyet (kadın/erkek/unisex) ya da yaş grubu (bebek/çocuk/genç/yetişkin) yazıyor
    mu, ya da ima ediyor mu?
      → HAYIR ise: cinsiyet/yaş bu satırda ölçüt değildir; ürün sorgunun istediği şeyse
        relevant, yakınsa semi-relevant, değilse irrelevant.
      → EVET ise S2.
S2. Ürünün cinsiyeti / yaş grubu sorgununkiyle aynı mı?
      → HAYIR ise: irrelevant. Ürün başka her yönden mükemmel uysa bile. Kadın ürünü erkek
        sorgusuna, çocuk ürünü yetişkin sorgusuna verilemez.
      → EVET ise: ürün ayrıca sorgunun istediği şeyin kendisiyse relevant, yakın bir
        varyantı/aksesuarıysa semi-relevant."""

ATTR_BRAND = u"""KARAR YÖNTEMİ — MARKA VE MODEL UYUMU ÜZERİNDEN

Bu satırda kararını ÖNCELİKLE marka/model/nesil uyumuna göre ver. Diğer her şey ikincildir.

S1. Sorgu bir marka, model, seri ya da nesil adı içeriyor mu? (samsung a54, iphone 15, pejo 307,
    airfryer XL gibi)
      → HAYIR ise: marka bu satırda ölçüt değildir; ürün sorgunun istediği şeyse relevant,
        yakınsa semi-relevant, değilse irrelevant.
      → EVET ise S2.
S2. Ürün o markanın/modelin kendisi mi, ya da açıkça onunla UYUMLU olduğu belirtilen bir ürün mü?
      → Kendisi ise: relevant.
      → Onunla uyumlu bir aksesuar/parça ise: semi-relevant.
      → BAŞKA bir markanın/modelin ürünü ise: irrelevant. Ürün tipi aynı olsa bile, hatta
        neredeyse aynı ürün olsa bile."""


def system_gender(order='label'):
    return _attr_system(ATTR_GENDER, order)


def system_brand(order='label'):
    return _attr_system(ATTR_BRAND, order)


# --- axis D: information quantity, not criteria ---------------------------------------------
# `tiny` is L7's TINY_BLOCK verbatim -- the shortest prompt anyone in this run wrote, and never
# run on the 31B. `mega` is the opposite pole on the same axis: every rubric block this run has
# produced, concatenated, so the model is given far more instruction than any single arm.
def system_tiny(order='label'):
    return L7.system_tiny()


def system_mega(order='label'):
    return u"\n\n".join([P.ROL, P.ETIKETLER, P.KARAR, P.KALIBRASYON,
                          u"EK KURALLAR — SINIR DURUMLARI", PX.KARAR_HARD, PX.KALIBRASYON_HARD,
                          u"EK KURALLAR — GENİŞ semi BANDI", KARAR_SOFT, KALIBRASYON_SOFT,
                          u"EK KURALLAR — ÜRÜN TİPİ KAPISI", KARAR_TYPE,
                          u"EK KURALLAR — ALICI GÖZÜYLE", P.UC_TEPKI, P.ALICI_KALIBRASYON,
                          P.out_block(order)])



# --- diversity batch 2: two more axes ------------------------------------------------------
# `color` completes axis C (a third disjoint blind spot alongside gender and brand).
# `en` is a NEW axis nobody in this run has touched: the rubric's LANGUAGE. Every prompt in this
# repo is Turkish while the items are Turkish, so instruction and data share a token
# distribution. An English rubric over Turkish rows forces the model across that boundary on
# every row, which should produce errors that no Turkish-rubric arm makes -- and under the
# coordinator's inverse relation, being worse alone is not a cost.
ATTR_COLOR = u"""KARAR YÖNTEMİ — RENK, DESEN VE GÖRÜNÜM ÜZERİNDEN

Bu satırda kararını ÖNCELİKLE renk / desen / görünüm uyumuna göre ver. Diğer her şey ikincildir.

S1. Sorgu bir renk (siyah, beyaz, kırmızı, ekru...), bir desen (çizgili, puantiyeli, çiçekli,
    ekose, düz...) ya da bir görünüm/kesim (uzun kollu, askılı, kapüşonlu, dar paça...) yazıyor mu?
      → HAYIR ise: bu satırda görünüm ölçüt değildir; ürün sorgunun istediği şeyse relevant,
        yakınsa semi-relevant, değilse irrelevant.
      → EVET ise S2.
S2. Ürünün rengi / deseni / kesimi sorgununkiyle aynı mı?
      → AYNI ise: relevant.
      → YAKIN ama tam aynı değilse (lacivert-siyah, çizgili-ekose gibi): semi-relevant.
      → AÇIKÇA FARKLI ise: irrelevant. Ürün tipi doğru olsa bile; alıcı yazdığı rengi/deseni
        istiyor."""


def system_color(order='label'):
    return _attr_system(ATTR_COLOR, order)


EN_BLOCK = u"""You are a search-relevance rater for Trendyol, a Turkish e-commerce marketplace.
You are given a Turkish search query and one product shown for that query. Decide how well the
product matches the query.

LABELS (choose exactly one)

- relevant: the product IS the thing the query asks for, and every condition written explicitly in
  the query (product type, brand, model, colour, gender, age group) holds for the product.
  Details the query does not mention are not conditions.

- semi-relevant: the product serves the same need only PARTLY. Two patterns, the first far more
  common: (a) right product family, unwanted variant -- the main product type matches but the
  package, set/single, size or a secondary attribute differs from what the buyer asked for;
  (b) accessory, spare part, case or complement of the queried product rather than the product
  itself.

- irrelevant: an explicitly written condition of the query is violated. In this dataset an
  irrelevant product is almost never a random unrelated item -- it is usually from the same
  department and shares words with the query, but is the WRONG THING: a different product type
  serving a different need, a contradicted colour or pattern, or the wrong brand, model, gender
  or age group.

DECISION RULE
1. List the conditions the query states explicitly. Most queries are shorter than three words and
   state only a product type.
2. If one of those conditions is directly contradicted by the product, answer irrelevant. You must
   be able to name which condition and what contradicts it.
3. Otherwise, if the product is the requested thing and all stated conditions hold, answer relevant.
4. Otherwise answer semi-relevant.

All three labels occur about equally often in this data.

OUTPUT -- write a single line of JSON and nothing else:
{"etiket": "semi-relevant"}
The "etiket" field must be exactly one of: relevant, semi-relevant, irrelevant"""


def system_en(order='label'):
    return EN_BLOCK


SYSTEMS = {
    'hard':   PX.system_hard,
    'zs':     P.system_zs,
    'tie':    system_tie,
    'ultra':  system_ultra,
    'shwide': system_shwide,
    'wide2':  system_wide2,
    'wide3':  system_wide3,
    'softtype': system_softtype,
    'mega': system_mega,
    'color': system_color,
    'en': system_en,
    'tiny': system_tiny,
    'brand': system_brand,
    'gender': system_gender,
    'relnarrow': system_relnarrow,
    'relwide': system_relwide,
    'harsher': system_harsher,
    'harsh': system_harsh,
    'soft':   system_soft,
    'wide':   system_wide,
    'type':   system_type,
    'l7soft': system_l7soft,
}

# Rubric HEADS for the per-category few-shot (`catfs`) path, which assembles its own system
# message around a category-specific shot block.  Each entry is the list of blocks that goes
# BEFORE the examples; `catfs` appends `ORNEK_BASLIK + shots` and the output block itself.
CATFS_HEADS = {
    'hard': [P.ROL, P.ETIKETLER, PX.KARAR_HARD, PX.KALIBRASYON_HARD],   # L9's best arm, 0.8825
    'zs':   [P.ROL, P.ETIKETLER, P.KARAR, P.KALIBRASYON],
    'soft': [P.ROL, P.ETIKETLER, KARAR_SOFT, KALIBRASYON_SOFT],
    'wide': [P.ROL, ETIKETLER_WIDE, KARAR_WIDE, KALIBRASYON_WIDE],
    'type': [P.ROL, P.ETIKETLER, KARAR_TYPE, KALIBRASYON_TYPE],
    'tie':  [P.ROL, P.ETIKETLER, PX.KARAR_HARD, KALIBRASYON_TIE],
    'ultra':[P.ROL, ETIKETLER_ULTRA, KARAR_ULTRA, KALIBRASYON_ULTRA],
    'wide2':[P.ROL, ETIKETLER_WIDE2, KARAR_WIDE2, KALIBRASYON_WIDE2],
    'wide3':[P.ROL, ETIKETLER_WIDE2, KARAR_WIDE3, KALIBRASYON_WIDE3],
    'softtype':[P.ROL, P.ETIKETLER, KARAR_SOFTTYPE, KALIBRASYON_SOFT],
    'harsh':  [P.ROL, P.ETIKETLER, KARAR_HARSH, KALIBRASYON_HARSH],
    'harsher':[P.ROL, ETIKETLER_HARSHER, KARAR_HARSH, KALIBRASYON_HARSH],
    'relnarrow':[P.ROL, P.ETIKETLER, P.KARAR, KALIBRASYON_RELNARROW],
    'relwide':[P.ROL, P.ETIKETLER, P.KARAR, KALIBRASYON_RELWIDE],
    'tiny':   [L7.TINY_BLOCK],
    'color':  [P.ROL, P.ETIKETLER, ATTR_COLOR],
}


# =============================================================================================
# 12) `champ` -- route 1: the `mega` trick applied to the BEST rubrics instead of contradictory ones.
#
# MEASURED PREMISE.  `mega` concatenated five mutually inconsistent rubrics into one 3,200-token
# prompt and scored 0.8719 with the run's 5th-best calibration -- the 31B averaged the
# instructions rather than being confused by them. That is prompt-ensembling inside a single pass,
# and it was demonstrated on the WORST possible ingredients. This arm uses the best ones:
#
#   `wide2_fs`  0.8933  best single arm measured anywhere today
#   `wide3_fs`  0.8927  second best; same gate 1, gate 2 narrowed the other way
#   L11 `bsemi` the two-class framing -- `relevant` removed so the whole rubric budget lands on
#               the semi/irrelevant boundary, which F136 says carries 69% of the unanimous errors
#
# `wide2` and `wide3` disagree about gate 2 ON PURPOSE and both are kept: that disagreement is
# exactly the ingredient `mega` proved this model averages rather than trips over. L11's binary is
# folded in as a SUB-QUESTION rather than a replacement, so `relevant` stays in the label space
# (the arm must still be a 3-class arm) while the boundary gets its own dedicated block.
KARAR_CHAMP_G2 = u"""KAPI 2 — tip FARKLI ise. İki ölçüt birden uygula ve ikisi de geçerse semi-relevant yaz.

  Ölçüt A (ihtiyaç): "Bu ürün, alıcının bu sorguyu yazarken karşılamak istediği ihtiyacı
    karşılar mı, ya da o ihtiyaç için değerlendirilebilecek bir alternatif mi?"
  Ölçüt B (somutluk): Bunu SOMUT bir ilişkiyle gösterebiliyor musun — ürün ya sorgudaki ürünün
    aksesuarı/parçası/kılıfı/filtresi/aparatıdır, ya da onun yerine geçebilecek, alıcının yan
    yana koyup "hangisini alsam" diyeceği doğrudan bir alternatiftir?

  → İKİSİ DE evet ise: semi-relevant.
  → Ölçüt B'yi gösteremiyorsan: irrelevant. "Aynı reyonda" ya da "işine yarayabilir" yetmez.
  Ölçülmüş: ürün tipi tutmadığında satırların yaklaşık yarısı irrelevant'tır, yani burada
  yazı-tura atıyorsun; semi-relevant'ı hak ettir."""

KARAR_CHAMP = KARAR_WIDE3.split(u"KAPI 2 —")[0] + KARAR_CHAMP_G2

# L11's binary, folded in as the tie-break sub-procedure rather than as the whole prompt.
IKILI_BLOK = u"""SINIR SORUSU — yalnızca semi-relevant ile irrelevant arasında kaldığında sor

Bu noktada ürünün sorgunun tam istediği şey OLMADIĞI zaten bellidir. Geriye tek bir soru kalır:
ürün alıcının işine YARAYABİLİR mi (semi-relevant), yoksa alıcının aramasında hiç görmek
istemeyeceği YANLIŞ bir ürün müdür (irrelevant)?

Sorgunun AÇIKÇA yazdığı bir şart üründe doğrudan çiğneniyor mu?
  - Çiğnenen şartı ve neyle çiğnendiğini tek cümleyle SÖYLEYEBİLİYORSAN → irrelevant.
    İhlal şu dört alandan birinde olmalıdır: ÜRÜN TİPİ ("mont" → kazak), RENK/DESEN
    ("yeşil elbise" → kırmızı elbise), MARKA/MODEL ("samsung a54 kılıfı" → iphone 13 kılıfı),
    CİNSİYET/YAŞ ("erkek kot pantolon" → kadın kot pantolon).
  - Söyleyemiyorsan → semi-relevant.

Sorguda YAZMAYAN hiçbir şey şart değildir. Sorgu renk demediyse rengin farklı olması ihlal
değildir; sorgu adet/ölçü demediyse paket farkı ihlal değildir.

En sık yapılan hata, işe yarayabilecek ürüne irrelevant demektir:
- Aksesuar, yedek parça, uyumlu ürün, tamamlayıcı → HER ZAMAN semi-relevant.
- Aynı ürün tipinin farklı varyantı (takım/tekil, set/tek parça, farklı ölçü, farklı paket,
  yakın akraba alt-tür) → HER ZAMAN semi-relevant.
- Sorgu yalnızca bir marka adıysa, o markanın herhangi bir ürünü → semi-relevant."""

KALIBRASYON_CHAMP = u"""KALİBRASYON

- Üç etiket de yaklaşık eşit sıklıktadır: satırların üçte biri relevant, üçte biri semi-relevant,
  üçte biri irrelevant. Cevaplarının dağılımı buna yakın olmalı.
- Kapı 1'i geçen (ürün tipi anlamca tutan) bir satıra irrelevant yazıyorsan neredeyse kesinlikle
  yanılıyorsun: ölçülmüş, o durumda irrelevant olma ihtimali ~%11.
- Kapı 2'de tersi geçerlidir: tip tutmuyorsa irrelevant ihtimali ~%48'dir.
- ⚠️ relevant'ı BOŞALTMA. Sorgunun yazdığı her şart tutuyorsa ve ürün istenen şeyin ta
  kendisiyse cevap relevant'tır; semi-relevant'a kaçma. semi'nin genişlemesi irrelevant'tan
  gelmelidir, relevant'tan DEĞİL.
- Kararsız kaldığında semi-relevant seç."""

CHAMP_HEAD = [P.ROL, ETIKETLER_WIDE2, KARAR_CHAMP, IKILI_BLOK, KALIBRASYON_CHAMP]


def system_champ(order='label'):
    return u"\n\n".join(CHAMP_HEAD + [P.out_block(order)])


# `champ2`: identical ingredients, different ORDER -- the boundary sub-question is promoted above
# the two gates. `mega` showed the model averages instruction blocks; if it also averages over
# their ORDER this is a null, and if it does not it is a second strong arm for free.
CHAMP2_HEAD = [P.ROL, ETIKETLER_WIDE2, IKILI_BLOK, KARAR_CHAMP, KALIBRASYON_CHAMP]


def system_champ2(order='label'):
    return u"\n\n".join(CHAMP2_HEAD + [P.out_block(order)])


SYSTEMS['champ'] = system_champ
SYSTEMS['champ2'] = system_champ2
CATFS_HEADS['champ'] = CHAMP_HEAD
CATFS_HEADS['champ2'] = CHAMP2_HEAD


# `champ3` -- `champ` with the two leaks it opened closed, on the measurement `champ_fs` produced.
#
# MEASURED: `champ_fs` = 2257/2354/2889 (L1 778) against `wide3_fs` 2569/2421/2510 (L1 158) and
# `wide2_fs` 2270/2455/2775 (L1 550). Folding L11's binary in cost calibration in a specific and
# predictable way: `bsemi` was written for rows ALREADY KNOWN not to be `relevant`, so its whole
# calibration paragraph pushes toward `semi` ("the most common error is calling a usable product
# irrelevant") with nothing pushing back -- and applied to all 7,500 rows that lands 389 rows over
# on `semi` and 146 under on `relevant`.
#
# The fix keeps the binary's DISCRIMINATION (the four named violation fields, the nameability
# test, the accessory/variant list) and drops its PRIOR (the one-sided "most common error" list
# and the unconditional "if undecided pick semi"), replacing them with the three-way count anchor.
# The `relevant` guard is also promoted from the calibration block into the boundary block, where
# the leak actually happens.
IKILI_BLOK_NEUTRAL = u"""SINIR SORUSU — yalnızca semi-relevant ile irrelevant arasında kaldığında sor

Bu noktada ürünün sorgunun tam istediği şey OLMADIĞI zaten bellidir. Tek soru kalır: ürün
alıcının işine YARAYABİLİR mi (semi-relevant), yoksa alıcının aramasında hiç görmek
istemeyeceği YANLIŞ bir ürün müdür (irrelevant)?

Sorgunun AÇIKÇA yazdığı bir şart üründe doğrudan çiğneniyor mu?
  - Çiğnenen şartı ve neyle çiğnendiğini tek cümleyle SÖYLEYEBİLİYORSAN → irrelevant.
    İhlal şu dört alandan birinde olmalıdır: ÜRÜN TİPİ ("mont" → kazak), RENK/DESEN
    ("yeşil elbise" → kırmızı elbise), MARKA/MODEL ("samsung a54 kılıfı" → iphone 13 kılıfı),
    CİNSİYET/YAŞ ("erkek kot pantolon" → kadın kot pantolon).
  - Söyleyemiyorsan → semi-relevant.

Sorguda YAZMAYAN hiçbir şey şart değildir: sorgu renk demediyse rengin farklı olması ihlal
değildir, sorgu adet/ölçü demediyse paket farkı ihlal değildir.

Bu soru YALNIZCA semi-relevant ile irrelevant arasında sorulur. Sorgunun yazdığı her şart
tutuyor ve ürün istenen şeyin ta kendisiyse bu soruya hiç gelmezsin — cevap relevant'tır.
İki taraf da eşit sıklıktadır: bu sınıra düşen satırların yaklaşık yarısı semi-relevant,
yarısı irrelevant'tır. Ne bol keseden semi-relevant dağıt, ne de her farkı ihlal say."""

CHAMP3_HEAD = [P.ROL, ETIKETLER_WIDE2, KARAR_CHAMP, IKILI_BLOK_NEUTRAL, KALIBRASYON_CHAMP]


def system_champ3(order='label'):
    return u"\n\n".join(CHAMP3_HEAD + [P.out_block(order)])


SYSTEMS['champ3'] = system_champ3
CATFS_HEADS['champ3'] = CHAMP3_HEAD
