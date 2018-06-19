from django.db import models

lang_codes = ['en', 'la', 'es', 'fr', 'it', 'ja', 'de', 'sv', 'da', 'cy', 'bg', 'pt', 'nl',
 'zh', 'el', 'he', 'ru', 'ko', 'pl', 'fi', 'eo', 'enm', 'sa', 'ale', 'yi', 'lt',
 'sr', 'no', 'ro', 'cs', 'tl', 'ca', 'is', 'myn', 'nai', 'ilo', 'ia', 'ga', 'fur',
 'af', 'kld', 'oc', 'nap', 'hu', 'fy', 'ceb', 'gl', 'nah', 'mi', 'nav', 'br',
 'arp', 'iu', 'bgi', 'gla', 'rmr', 'sl', 'te', 'oji', 'ar', 'et', 'ang', 'fa',]
    
lang_lookup = lambda x: lang_codes.index(x) if x in lang_codes else -1

class Author(models.Model):
    fullname = models.CharField(max_length=100)
    note = models.CharField(max_length=100,null=True, blank=True)
    gutenberg_id = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.fullname
    
class Book(models.Model):
    title = models.CharField(max_length=200)
    pickle_object = models.FilePathField(null=True, blank=True)
    hash = models.CharField(max_length=200)
    authors = models.ManyToManyField(Author)
    language = models.CharField(max_length=100)
    lang_code_id = models.IntegerField(default=-1)
    downloads = models.IntegerField(default=0)
    
    from_gutenberg = models.BooleanField(default=False)
    gutenberg_id = models.IntegerField(null=True, blank=True)
    mobi_file_path = models.FilePathField(null=True, blank=True)
    epub_file_path = models.FilePathField(null=True, blank=True)
    txt_file_path = models.FilePathField(null=True, blank=True)
    expanded_folder_path = models.FilePathField(null=True, blank=True)
    
    # more basic info
    length = models.IntegerField(default=0)
    numUniqWords = models.IntegerField(default=0)
    ignorewords = models.CharField(max_length=400,
                                   help_text="Comma separated list of words to ignore from this one.")
    wiki = models.URLField(null=True, blank=True)
    scaling_exponent = models.FloatField(null=True, blank=True,
        help_text="Zipf law fit scaling exponent.")
    scaling_exponent_top100 = models.FloatField(null=True, blank=True,
        help_text="The scaling exponent fit across just the top 100.")

    # if we had an issue processing it....
    exclude = models.BooleanField(default=False)
    excludeReason = models.CharField(max_length=100, null=True, blank=True)

    # library of congress class
    locc_string = models.CharField(max_length=5, null=True, blank=True)
    locc_with_P = models.BooleanField(default=False)

    def __str__(self):
        return self.title

