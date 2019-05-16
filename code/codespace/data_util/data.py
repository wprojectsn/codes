import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'



class Vocab(object):

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 

    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          print 'Warning: incorrectly formatted line in vocabulary file: %s\n' % line
          continue
        w = pieces[0]
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print "max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count)
          break

    print "Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1])

  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    return self._count

  def write_metadata(self, fpath):
    print "Writing word embedding metadata file to %s..." % (fpath)
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in xrange(self.size()):
        writer.writerow({"word": self._id_to_word[i]})


class Concept_vocab(object):

  def __init__(self, vocab_file, max_size):
    self._id_to_word = {}
    self._id_to_p = {}
    self._count = 0

    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 10:
          print 'Warning: incorrectly formatted line in vocabulary file: %s\n' % line
          continue
        w = []
        p = []
        w1 = pieces[0]
        w2 = pieces[2]
        p1 = pieces[1]
        p2 = pieces[3]
        w.append(w1)
        w.append(w2)
        p.append(p1)
        p.append(p2)
        self._id_to_word[self._count] = w
        self._id_to_p[self._count] = p
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print "max_size of concept_vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count)
          break

    print "Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1])

    
  '''def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]'''

  def id2word(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]
    
  def id2p(self, word_id):
    return self._id_to_p[word_id]
    
  def size(self):
    return self._count
        
        
def example_generator(data_path, single_pass):
  while True:
    filelist = glob.glob(data_path)
    assert filelist, ('Error: Empty filelist at %s' % data_path)
    if single_pass:
      filelist = sorted(filelist)
    else:
      random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)
    if single_pass:
      print "example_generator completed reading all datafiles. No more data."
      break

def getprobase(word):
  try:
    response = urllib.urlopen('https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance='+word+'&topK=10')
    html = eval(response.read())
    if html:
      maxw = []
      maxp = []
      mask = []
      w_num = 0
      concept = sorted(html.items(), key=lambda item: item[1], reverse = True)
      for i in range(10):
        maxw_i = concept[i][0]
        maxp_i = concept[i][1]
        if len(maxw_i.split()) == 2:
          maxw.append(maxw_i)
          maxp.append(maxp_i)
          maxk.append(1)
        else:
          maxw.append('UNK')
          maxp.append(0)
          maxk.append(0)
        w_num += 1
        if w_num == 2:
          break
    else:
      maxw = ['UNK','UNK']
      maxp = [0,0]
      mask = [0,0]
  except:
      maxw = ['UNK','UNK']
      maxp = [0,0]
      mask = [0,0]
  return maxw, maxp, mask

def article2ids(article_words, vocab, concept_vocab):
  ids = []
  oovs = []
  concept_ids = []
  concept_p = []
  concept_mask = []
  position = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in article_words:
    single_ids = []
    single_p =[]
    single_mask = []
    single_position = []
    i = vocab.word2id(w)
    if i == unk_id:
      if w not in oovs:
        oovs.append(w)
      oov_num = oovs.index(w)
      ids.append(vocab.size() + oov_num)
      concept_word, p, mask = getprobase(w)
      for word_i in concept_word:
        j = vocab.word2id(word_i)
        single_position.append(j)
        if j == unk_id:
          if word_i not in oovs:
            oovs.append(word_i)
          oov_num = oovs.index(word_i)
          single_ids.append(vocab.size() + oov_num)
          single_p = p
          single_mask = mask
          
        else:
          single_ids.append(j)
          single_p = p
          single_mask = mask
    else:
      ids.append(i)
      concept_word = concept_vocab.id2word(i)
      for word_i in concept_word:
        if word_i == 'UNK':
          single_mask.append(0)
        else:
          single_mask.append(1)
        j = vocab.word2id(word_i)
        single_position.append(j)
        if j == unk_id:
          if word_i not in oovs:
            oovs.append(word_i)
          oov_num = oovs.index(word_i)
          single_ids.append(vocab.size() + oov_num)
        else:
          single_ids.append(j)
      p = concept_vocab.id2p(i)
      single_p = p
    concept_ids.append(single_ids)
    concept_p.append(single_p)
    position.append(single_position)
    concept_mask.append(single_mask)
  return ids, oovs, concept_ids, concept_p, position, concept_mask


def abstract2ids(abstract_words, vocab, article_oovs):
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in abstract_words:
    i = vocab.word2id(w)
    if i == unk_id:
      if w in article_oovs:
        vocab_idx = vocab.size() + article_oovs.index(w)
        ids.append(vocab_idx)
      else:
        ids.append(unk_id)
    else:
      ids.append(i)
  return ids


def outputids2words(id_list, vocab, article_oovs):
  words = []
  for i in id_list:
    try:
      w = vocab.id2word(i)
    except ValueError as e:
      assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
      article_oov_idx = i - vocab.size()
      try:
        w = article_oovs[article_oov_idx]
      except ValueError as e:
        raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
    words.append(w)
  return words


def abstract2sents(abstract):
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(SENTENCE_START, cur)
      end_p = abstract.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(abstract[start_p+len(SENTENCE_START):end_p])
    except ValueError as e:
      return sents
      
def article2sents(article):
  cur = 0
  sents = []
  while True:
    try:
      start_p = article.index(SENTENCE_START, cur)
      end_p = article.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(article[start_p+len(SENTENCE_START):end_p])
    except ValueError as e:
      return sents


def show_art_oovs(article, vocab):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = article.split(' ')
  words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
  out_str = ' '.join(words)
  return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = abstract.split(' ')
  new_words = []
  for w in words:
    if vocab.word2id(w) == unk_token:
      if article_oovs is None:
        new_words.append("__%s__" % w)
      else:
        if w in article_oovs:
          new_words.append("__%s__" % w)
        else:
          new_words.append("!!__%s__!!" % w)
    else:
      new_words.append(w)
  out_str = ' '.join(new_words)
  return out_str
