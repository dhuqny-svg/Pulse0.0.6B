import numpy as np
import random
import json
import os

def load_vocab(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy {file_path}, dùng từ mặc định")
        return ["chào", "bạn", "tôi", "là", "có", "không", "và", "với", "đi", "ăn", "ngủ"]
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        return list(vocab_dict.keys())
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

class VocabViet:
    def __init__(self, file_path='vocab.txt'):
        words = load_vocab(file_path)
        special = ["<PAD>", "<UNK>", "<START>", "<END>"]
        self.words = special + list(dict.fromkeys(words))
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.words)
        print(f"📚 Từ vựng: {self.vocab_size} từ")
    
    def encode(self, text):
        return [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in text.split()]
    
    def decode(self, ids):
        return " ".join([self.idx2word.get(i, "<UNK>") for i in ids])

class TransformerLM:
    def __init__(self, vocab, d_model=128, num_heads=4, num_layers=3):
        self.vocab = vocab
        self.d_model = d_model
        self.num_heads = num_heads
        self.embedding = np.random.randn(vocab.vocab_size, d_model) * 0.02
        self.pos_encoding = self._pos_encoding(512, d_model)
        self.W_out = np.random.randn(d_model, vocab.vocab_size) * 0.02
        self.b_out = np.zeros(vocab.vocab_size)
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(self._create_layer())
        
        # Tải config generation nếu có
        self.gen_config = self._load_generation_config()
    
    def _load_generation_config(self, config_path='generation_config.json'):
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                print("✅ Đã tải generation_config.json")
                return cfg
        # Mặc định
        print("⚠️ Không có generation_config.json, dùng mặc định")
        return {
            "max_length": 20,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "do_sample": True
        }
    
    def _pos_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        pe[:, 0::2] = np.sin(pos / 10000 ** (2 * (i // 2) / d_model))[:, 0::2]
        pe[:, 1::2] = np.cos(pos / 10000 ** (2 * (i // 2) / d_model))[:, 1::2]
        return pe
    
    def _create_layer(self):
        return {
            'W_q': np.random.randn(self.d_model, self.d_model) * 0.02,
            'W_k': np.random.randn(self.d_model, self.d_model) * 0.02,
            'W_v': np.random.randn(self.d_model, self.d_model) * 0.02,
            'W_o': np.random.randn(self.d_model, self.d_model) * 0.02,
            'W1': np.random.randn(self.d_model, self.d_model * 4) * 0.02,
            'b1': np.zeros(self.d_model * 4),
            'W2': np.random.randn(self.d_model * 4, self.d_model) * 0.02,
            'b2': np.zeros(self.d_model),
        }
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / (np.sum(e_x) + 1e-10)
    
    def _attention(self, Q, K, V):
        scores = Q @ K.T / np.sqrt(self.d_model)
        probs = self._softmax(scores)
        return probs @ V
    
    def forward(self, ids):
        seq_len = len(ids)
        x = self.embedding[ids] + self.pos_encoding[:seq_len]
        for layer in self.layers:
            Q = x @ layer['W_q']
            K = x @ layer['W_k']
            V = x @ layer['W_v']
            attn = self._attention(Q, K, V) @ layer['W_o']
            x = x + attn
            ff = np.maximum(0, x @ layer['W1'] + layer['b1'])
            ff = ff @ layer['W2'] + layer['b2']
            x = x + ff
        return x @ self.W_out + self.b_out
    
    def generate(self, start_text):
        cfg = self.gen_config
        max_tokens = cfg.get("max_length", 20)
        temp = cfg.get("temperature", 0.8)
        top_k = cfg.get("top_k", 50)
        top_p = cfg.get("top_p", 0.9)
        penalty = cfg.get("repetition_penalty", 1.0)
        do_sample = cfg.get("do_sample", True)
        
        ids = self.vocab.encode(start_text)
        generated = []
        
        for _ in range(max_tokens):
            logits = self.forward(ids)
            probs = self._softmax(logits[-1] / temp)
            
            # Áp dụng top_k
            if top_k > 0 and top_k < len(probs):
                top_k_indices = np.argsort(probs)[-top_k:]
                mask = np.zeros_like(probs)
                mask[top_k_indices] = 1
                probs = probs * mask
                probs = probs / np.sum(probs)
            
            # Áp dụng top_p (nucleus)
            if top_p < 1.0:
                sorted_indices = np.argsort(probs)[::-1]
                cumulative = np.cumsum(probs[sorted_indices])
                cutoff = np.searchsorted(cumulative, top_p) + 1
                mask = np.zeros_like(probs)
                mask[sorted_indices[:cutoff]] = 1
                probs = probs * mask
                probs = probs / np.sum(probs)
            
            # Áp dụng repetition penalty
            if penalty != 1.0:
                for token in set(ids):
                    if token in self.vocab.word2idx.values():
                        probs[token] = probs[token] ** penalty
            
            # Chọn từ tiếp theo
            if do_sample:
                next_id = np.random.choice(len(probs), p=probs)
            else:
                next_id = np.argmax(probs)
            
            if next_id == self.vocab.word2idx.get("<END>", -1):
                break
            
            ids.append(next_id)
            generated.append(next_id)
        
        return self.vocab.decode(ids)

DATA_MAU = [
    "Ê mày khỏe không",
    "Trời ơi lâu quá không gặp",
    "Dạo này cuộc sống thế nào rồi",
    "Ủa hôm nay mày rảnh hả",
    "Chào chị em mới vào công ty tuần trước",
    "A lô ai đấy ạ",
    "Xin lỗi tôi có thể ngồi đây được không",
    "Cháu chào cô ạ cô khỏe không ạ",
    "Làm ơn chỉ giúp tôi đường ra bến xe với",
    "Mày xem giúp tao cái này với tao chẳng hiểu gì cả",
    "Chị ơi cho em hỏi cái máy này dùng sao ạ",
    "Anh ơi bưng giúp em cái này với nặng quá",
    "Cho em mượn cái bút với em quên mang rồi",
    "Tối nay qua nhà tao ăn cơm nhé tao nấu món mới",
    "Hôm nay mệt quá đi mất chẳng muốn làm gì cả",
    "Mày ơi tao vừa chia tay rồi buồn vãi",
    "Trời ơi vui quá tao được tăng lương rồi",
    "Chán đời thật sự luôn ấy chẳng biết làm gì",
    "Hôm qua tao cãi nhau với mẹ giờ vẫn còn ấm ức",
    "Yêu đương gì mà mệt mỏi thế này tao phát ngán rồi",
    "Cảm ơn mày nhé có mày bên cạnh tao đỡ tủi thân",
    "Tự dưng hôm nay tao thấy cô đơn thế nào ấy",
    "Có chuyện gì vui kể tao nghe với tao đang chán",
    "Mày ăn cơm chưa đói không",
    "Sao mặt mũi lại thế có chuyện gì à",
    "Mày ngủ bao nhiêu tiếng hôm qua mắt thâm đen kìa",
    "Dạo này mẹ mày có khỏe không",
    "Công việc dạo này ổn định chứ có áp lực gì không",
    "Ủa sao hôm nay mày đi làm sớm thế",
    "Cái này giá bao nhiêu vậy chị",
    "Mày biết tin gì chưa nghe nói thằng Hùng sắp lấy vợ",
    "Trời ơi sao cứ mỗi lần ăn xong là tao lại buồn ngủ thế",
    "Bạn có biết quán này mở cửa đến mấy giờ không",
    "Mày có thấy con bé đó xinh không hay tao nhìn nhầm",
    "Sao ngày hôm nay trời nóng bất thường thế này",
    "Làm thế nào để hết bị muỗi đốt vậy trời",
    "Cậu có biết cuốn sách đó tên gì không tao tìm mãi",
    "Cuối tuần này đi chơi đâu không tao chán ở nhà quá",
    "Mày xem phim gì hay chưa giới thiệu tao với",
    "Tao mới mua được con game hay lắm tối qua chơi tới 3h sáng",
    "Hay mai mình đi ăn lẩu đi trời lạnh thế này",
    "Mày có thích uống cà phê không tao biết quán ngon lắm",
    "Dạo này tao nghiện xem TikTok chẳng làm được gì cả",
    "Mày tập gym chưa cho tao đi cùng với",
    "Mày thích thể loại nhạc gì tao đang cày bài của Sơn Tùng",
    "Mua quần áo mới chưa đi Mall với tao đi",
    "Trời ơi bài tập nhiều vãi làm không xuể",
    "Sếp giao việc nhiều quá không thở nổi luôn",
    "Lương tháng này ít quá tiêu không đủ",
    "Nhà trọ lại bị cắt nước rồi chán không",
    "Học hành thì nhiều đầu óc thì rối bù",
    "Trời mưa thế này đi làm sao đây trời ơi",
    "Đường tắc quá tao đi làm muộn mất rồi",
    "Cơm trưa ở cty nhạt nhẽo chả muốn ăn gì",
    "Sao cứ hết tiền vào giữa tháng thế nhỉ không hiểu nổi",
    "Mày nhìn mặt tao xem có giống thằng ngốc không",
    "Trời ơi tao quên mất sinh nhật nó rồi chết thật",
    "Hôm qua tao đi chợ mua tới 3 mớ rau mà chỉ có 1 mớ mang về",
    "Mày mà làm thế thì tao gọi là tính mạng treo trên sợi chỉ đấy",
    "Tao vừa bị con mèo nó chửi cho một trận đời vẫn đẹp sao",
    "Có ai đẹp trai như tao mà vẫn ế không nhỉ",
    "Mày ơi tao mới phát hiện ra nấu mì tôm cũng cần có tài năng",
    "Học ngoại ngữ tao chỉ giỏi mỗi câu I love you thôi",
    "Nếu mày gặp rồng mày sẽ hỏi nó câu gì",
    "Tao vừa ăn no quá giờ mà có ai bảo chạy marathon chắc tao khóc",
    "Trời ơi sao mà tệ vậy nè",
    "Mày làm cái gì mà giờ này mới về hả",
    "Im đi tao muốn yên tĩnh một tí",
    "Cút đi cho tao nhờ đang bực mình đấy",
    "Không nghe tao nói à tao bảo dừng lại",
    "Chịu thôi tao bó tay với nó rồi",
    "Lần sau đừng có làm thế nữa tao không thích",
    "Ủa mày mới nói cái gì thế nói lại coi",
    "Đừng có đùa như thế tao không vui đâu",
    "Tao mệt mỏi với mấy trò này lắm rồi",
    "Tao nên mua xe máy hay ô tô bây giờ",
    "Làm sao để con gái thích mình nhỉ mày có kinh nghiệm không",
    "Tao có nên nghỉ việc không làm mãi chẳng ra gì",
    "Có nên sống chung với bạn gái lúc này không",
    "Bạn thân của tao sắp lấy chồng tao nên tặng gì",
    "Bố mẹ phản đối tao phải làm sao để thuyết phục",
    "Học đại học có thực sự cần thiết không",
    "Làm gì với số tiền tiết kiệm ít ỏi này",
    "Có nên cho người yêu mượn tiền không",
    "Hôm qua tao đi chơi gặp một cô gái xinh như mơ",
    "Trời hôm nay đẹp quá đứng dưới cây hoa đào mà lòng man mác",
    "Nó đến gặp tao mặt mày bơ phờ chẳng nói chẳng rằng",
    "Tao đang ngồi uống trà thì thấy một con mèo nó nhìn tao chằm chằm",
    "Ngồi trên ban công gió thổi nhẹ tự dưng tao thấy lâng lâng",
    "Bữa cơm tối qua cả nhà quây quần vui cực kỳ",
    "Đi dọc con phố tao gặp toàn người quen cứ phải chào mãi",
    "Nó bảo tao chờ tao tí thế mà 2 tiếng sau nó mới đến",
    "Con bé nhà hàng xóm hôm nay mặc váy đỏ đi thi đáng yêu lắm",
    "Của bền tại người mày ạ",
    "Chậm mà chắc còn hơn nhanh mà vỡ mồm",
    "Ăn quả nhớ kẻ trồng cây thế thôi",
    "Có công mài sắt có ngày nên kim mà",
    "Lửa thử vàng gian nan thử sức",
    "Ở hiền gặp lành tao luôn tin thế",
    "Trăm hay không bằng tay quen mày tập đi rồi sẽ giỏi",
    "Đi một ngày đàng học một sàng khôn",
    "Nước chảy đá mòn từ từ rồi sẽ ổn",
    "Thùng rỗng kêu to đừng có mà ba hoa",
    "Nhớ hồi đó tụi mình đi chơi suốt giờ chẳng gặp nữa",
    "Tao nhớ mãi cái ngày đầu tiên đi học khóc như mưa",
    "Chuyện xưa mà nhắc lại vẫn thấy buồn",
    "Tao với nó từng thân nhau lắm giờ chẳng biết nó thế nào",
    "Ngày đó nghèo mà vui bây giờ có tiền lại thấy trống rỗng",
    "Hồi học cấp 3 toàn trốn học đi chơi game",
    "Tao còn giữ tấm hình ngày ấy dù nó đã cũ mèm",
    "Mày còn nhớ cái vụ tai nạn năm 2018 không",
    "Giá như ngày đó tao không nói câu ấy",
    "Mày nghĩ ai sẽ thắng bầu cử sắp tới",
    "Giá vàng hôm nay là bao nhiêu nhỉ",
    "Nghề nào lương cao nhất hiện nay",
    "Theo mày Covid có thật sự biến mất không",
    "Bảo hiểm y tế dùng thế nào tao chẳng biết gì",
    "Có nên cho con học trường quốc tế không",
    "Cách phân biệt hàng thật hàng giả",
    "Mua nhà ở Hà Nội bây giờ có giá bao nhiêu",
    "Đi xuất khẩu lao động có sướng không",
    "Mày ơi tao vừa nhặt được cái này không biết là gì",
    "Trời ơi con gì lạ vậy nó bò kìa",
    "Đừng nói là mày không tin vào ma nhé",
    "Nếu mày là tớ mày sẽ làm gì",
    "Mày có từng nghĩ đến việc bỏ hết mọi thứ chưa",
    "Nói thật nhé tao đã làm mất sợi dây chuyền của mẹ",
    "Tao đang xem lại bộ phim cũ mày xem cùng không",
    "Sao hôm nay trông mày trẻ ra thế tiêm botox à",
    "Nhiều khi tự dưng tao thấy mình thật ngốc",
    "Bạn tên là gì",
    "Hôm nay thời tiết thế nào",
    "Bạn có khỏe không",
    "Bạn thích ăn món gì",
    "Bạn làm được những gì",
    "Tôi nên bắt đầu từ đâu",
    "Bạn có biết tôi không",
    "Bạn có rảnh không",
    "Bạn đến từ đâu",
    "Bạn có cảm xúc không",
    "Ngày hôm nay của bạn thế nào",
    "Bạn thích làm gì khi rảnh",
    "Bạn có gia đình không",
    "Bạn sống ở đâu",
    "Bạn học trường nào",
    "Bạn làm nghề gì",
    "Tôi cảm thấy hơi buồn",
    "Làm sao để vượt qua nỗi sợ hãi",
    "Bạn nghĩ gì về tình yêu",
    "Tôi nên chọn nghề nào",
    "Tại sao tôi lại hay trì hoãn",
    "Bạn có lời khuyên gì cho tôi không",
    "Làm thế nào để tôi tự tin hơn",
    "Bạn có bao giờ nói dối không",
    "Tôi đang rất căng thẳng",
    "Tôi cảm thấy mình thất bại",
    "Làm sao để tôi vui lên",
    "Tôi có nên thay đổi không",
    "Bạn nghĩ gì về cái chết",
    "Tôi sợ cô đơn",
    "Làm sao để tha thứ cho người khác",
    "Tôi muốn tìm lại chính mình",
    "Tôi vừa cãi nhau với sếp phải làm sao",
    "Làm thế nào để bắt đầu một cuộc trò chuyện với người lạ",
    "Con tôi không chịu học bài",
    "Hàng xóm nuôi chó sủa cả đêm",
    "Làm sao để quên người yêu cũ",
    "Bạn tôi nghiện game khuyên thế nào",
    "Tôi muốn tỏ tình với crush nói sao đây",
    "Làm sao để đàm phán giá cả",
    "Tôi bị quên mật khẩu",
    "Làm thế nào để vượt qua kỳ thi",
    "Tôi muốn từ bỏ công việc hiện tại",
    "Làm sao để chọn quà cho mẹ",
    "Tôi muốn đi du lịch một mình lời khuyên",
    "Làm thế nào để học một kỹ năng mới",
    "Tục lệ cúng ông Công ông Táo là gì",
    "Ngày Valentine có ý nghĩa gì",
    "Tại sao người ta lại uống cà phê",
    "Lễ hội Halloween bắt nguồn từ đâu",
    "Tại sao người Việt ăn Tết Nguyên Đán",
    "Bạn có thích xem bóng đá không",
    "Thời trang đường phố bây giờ thế nào",
    "Có nên sống thử trước khi kết hôn",
    "Vai trò của phụ nữ trong xã hội hiện đại",
    "Làm thế nào để bảo vệ môi trường",
    "Tại sao giới trẻ thích mạng xã hội",
    "Bạn nghĩ sao về việc kinh doanh online",
    "Lợi ích của việc đọc sách là gì",
    "Ca sĩ bạn yêu thích là ai",
    "Bộ phim nào nên xem nhất",
    "Cuốn sách nào nên đọc",
    "Bạn có tin vào tình yêu qua mạng không",
    "Làm sao để quản lý thời gian hiệu quả",
    "Kỹ năng mềm quan trọng nhất là gì",
    "Cách viết CV xin việc ấn tượng",
    "Làm sao để thuyết trình tự tin",
    "Bạn có biết lập trình không dạy tôi",
    "Cách giải quyết xung đột trong nhóm",
    "Làm thế nào để tăng năng suất làm việc",
    "Bí quyết để được tăng lương",
    "Tôi nên học thêm kỹ năng gì trong năm nay",
    "Làm sao để xây dựng thương hiệu cá nhân",
    "Cách đặt câu hỏi thông minh",
    "Làm sao để trở thành nhà lãnh đạo giỏi",
    "Cách phát triển tư duy phản biện",
    "Bạn nghĩ AI sẽ thay thế con người không",
    "Làm thế nào để làm việc nhóm hiệu quả",
]

def train_transformer(model, data, epochs=50, lr=0.001):
    print(f"🧠 Huấn luyện {len(data)} câu, {epochs} epochs")
    for epoch in range(epochs):
        total_loss = 0
        for text in data:
            ids = model.vocab.encode(text)
            if len(ids) < 2:
                continue
            logits = model.forward(ids[:-1])
            target = ids[-1]
            probs = model._softmax(logits[-1])
            loss = -np.log(probs[target] + 1e-10)
            total_loss += loss
            grad = probs.copy()
            grad[target] -= 1
            model.embedding[ids[:-1]] += lr * grad.mean() * 0.01
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch}: loss = {total_loss:.4f}")
    print("✅ Huấn luyện xong!")

class PulseChatbot:
    def __init__(self, vocab_file='vocab.txt'):
        self.vocab = VocabViet(vocab_file)
        self.model = TransformerLM(self.vocab)
    
    def train(self, data, epochs=50):
        train_transformer(self.model, data, epochs)
        return self
    
    def chat(self, text):
        if not text:
            return "Nói gì đi bạn!"
        return self.model.generate(text)

if __name__ == "__main__":
    print("="*60)
    print("🚀 PULSE TRANSFORMER - ĐẦY ĐỦ (VOCAB + CONFIG)")
    print("="*60)
    
    bot = PulseChatbot('vocab.txt')
    bot.train(DATA_MAU, epochs=50)
    
    print("\n" + "="*60)
    print("🧪 TEST CHAT:")
    print("="*60)
    
    test_cases = [
        "chào bạn",
        "hôm nay thế nào",
        "mày khỏe không",
        "rảnh không",
        "tôi buồn quá",
        "tạm biệt",
    ]
    
    for q in test_cases:
        r = bot.chat(q)
        print(f"👤 {q}")
        print(f"🤖 {r}")
        print()
    
    print("="*60)
    print("✅ PULSE TRANSFORMER ĐÃ SẴN SÀNG!")
    print("💡 Gõ 'exit' để thoát")
    print("="*60)
    
    while True:
        user_input = input("\n👤 Bạn: ").strip()
        if user_input.lower() in ["exit", "quit", "thoát"]:
            print("🤖 Pulse: Tạm biệt! Hẹn gặp lại!")
            break
        if user_input:
            response = bot.chat(user_input)
            print(f"🤖 Pulse: {response}")